# [MODULE DOC] Experimental FX quantization demo used only for testing. Defines observer
# classes that record min/max statistics while interpreting a traced graph, a registry
# mapping patterns (modules, functions or nested tuples of them) to handler classes, and a
# Quantizer that matches patterns on a GraphModule, observes sample inputs, and builds a new
# quantized GraphModule. Import-time side effect: the decorated handler classes fill the
# module-level pattern registry.
r"""
**This file is EXPERIMENTAL and is mostly used for testing purposes! Do not
rely on it for anything!**
"""

import operator
import sys

import torch
from torch.fx import Graph, GraphModule
from torch.fx.graph import map_arg
from torch.fx.proxy import Proxy
from torch.nn.utils import fuse_conv_bn_weights


# can be a
#  module type, a builtin function, or a string to match target


def _minmax_scale_zeropoint(
    min_val, max_val, qmin=-127, qmax=128, eps=torch.finfo(torch.float32).eps
):
    # [FUNCTION DOC] HELPER: Computes an affine quantization scale and zero point from a
    # value range. The range is first widened to include zero. If it is degenerate, returns
    # scale 1.0 and zero point 0. Otherwise scale is the range divided by the quantized
    # span, floored at eps, and the zero point is derived from the minimum, clamped into the
    # quantized range and converted to int. Returns the pair (scale, zero point). The eps
    # default is the float32 machine epsilon, evaluated once at definition time.
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)
    if max_val == min_val:
        return 1.0, 0
    else:
        scale = (max_val - min_val) / float(qmax - qmin)
        scale = max(scale, eps)
        zero_point = qmin - round(min_val / scale)
        zero_point = max(qmin, zero_point)
        zero_point = min(qmax, zero_point)
        zero_point = int(zero_point)
        return scale, zero_point


class MinMaxObserver:
    # [CLASS DOC] HELPER: Observer handler that tracks the running minimum and maximum of
    # the tensor values produced by a matched node. State: min and max (start at infinity
    # and negative infinity) and all_tensors, which becomes False once a non-tensor value is
    # seen.
    def __init__(self, quantizer, node):
        # [FUNCTION DOC] HELPER: Initializes min to positive infinity, max to negative
        # infinity and all_tensors to True. The quantizer and node arguments are accepted
        # for interface compatibility and ignored.
        self.min, self.max = float("inf"), float("-inf")
        self.all_tensors = True

    def observe(self, node, env):
        # [FUNCTION DOC] HELPER: Records the value computed for the node (looked up by node
        # name in the environment mapping). A non-tensor value clears all_tensors and is
        # otherwise ignored; a tensor widens the stored min and max.
        v = env[node.name]
        if not isinstance(v, torch.Tensor):
            self.all_tensors = False
            return
        self.max = max(self.max, float(v.max()))
        self.min = min(self.min, float(v.min()))

    def scale_zeropoint(self):
        # [FUNCTION DOC] HELPER: Returns the (scale, zero point) pair for the observed
        # range, using the unsigned 8-bit span from 0 to 255.
        return _minmax_scale_zeropoint(self.min, self.max, qmin=0, qmax=255)


class NoObserver:
    # [CLASS DOC] HELPER: Handler base class for patterns that need no statistics; both its
    # constructor and its observe method do nothing.
    def __init__(self, quantizer, node):
        # [FUNCTION DOC] HELPER: Does nothing; arguments are ignored.
        pass

    def observe(self, node, env):
        # [FUNCTION DOC] HELPER: Does nothing; observation is a no-op for this handler type.
        pass


_DEFAULT_QUANTIZATION_PATTERNS = {}


def register_pattern(pattern):
    # [FUNCTION DOC] Decorator factory for the pattern registry. Given a pattern, returns a
    # decorator that stores the decorated handler class in the module-level default patterns
    # mapping under that pattern and returns the class unchanged.
    def insert(fn):
        # [FUNCTION DOC] HELPER: Registers the given function in the default patterns
        # mapping under the enclosing pattern (mutating module state) and returns it
        # unchanged.
        _DEFAULT_QUANTIZATION_PATTERNS[pattern] = fn
        return fn

    return insert


@register_pattern(operator.add)
class Add(MinMaxObserver):
    # [CLASS DOC] HELPER: Handler for the add operator pattern, observing min/max of the
    # sum.
    def quantize(self, quantizer, node, load_arg):
        # [FUNCTION DOC] HELPER: Produces a quantized add. If any non-tensor value was
        # observed returns NotImplemented so the quantizer copies the node unchanged.
        # Otherwise creates a function-call node in the quantizer's new graph targeting the
        # quantized add op, with args loaded through the supplied load_arg callback and the
        # observed scale and zero point as keyword arguments. Returns that node.
        if not self.all_tensors:
            return NotImplemented
        scale, zeropoint = self.scale_zeropoint()
        return quantizer.quantized_graph.create_node(
            "call_function",
            torch.ops.quantized.add,
            load_arg(node.args),
            {"scale": scale, "zero_point": zeropoint},
        )


class Relu(NoObserver):
    # [CLASS DOC] HELPER: Handler for relu that needs no observation. Not registered in the
    # pattern registry in this file.
    def quantize(self, quantizer, node, load_arg):
        # [FUNCTION DOC] HELPER: Applies torch.relu directly to the loaded first argument of
        # the node and returns the resulting value (it does not create a graph node).
        return torch.relu(
            load_arg(node.args[0])
        )  # torch.relu works directly on quantized tensors?


# these ops have quantized equivalents that do not need any extra information
@register_pattern(torch.nn.ReLU)
@register_pattern(torch.nn.AvgPool2d)
@register_pattern(torch.nn.MaxPool2d)
@register_pattern(torch.nn.AdaptiveAvgPool2d)
class CopyNode(NoObserver):
    # [CLASS DOC] HELPER: Handler registered for ReLU, AvgPool2d, MaxPool2d and
    # AdaptiveAvgPool2d modules, whose quantized form needs no extra information. No
    # observation is done.
    def quantize(self, quantizer, node, load_arg):
        # [FUNCTION DOC] HELPER: Copies the node into the quantizer's new graph, with its
        # arguments mapped through load_arg, and returns the new node.
        return quantizer.quantized_graph.node_copy(node, load_arg)


class IdentityModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Pass-through module whose forward returns its input. Used as a
    # stand-in for batch norm modules that were fused away.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input unchanged.
        return x


# handle conv, maybe followed by bn, maybe followed by relu
@register_pattern(torch.nn.modules.conv.Conv2d)
@register_pattern((torch.nn.ReLU, torch.nn.modules.conv.Conv2d))
@register_pattern(
    (torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.conv.Conv2d)
)
@register_pattern(
    (
        torch.nn.ReLU,
        (torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.conv.Conv2d),
    )
)
class ConvNormRelu(MinMaxObserver):
    # [CLASS DOC] HELPER: Handler for a 2d convolution optionally followed by batch norm and
    # then relu, registered for the four combinations. State: relu_node and bn_node (None
    # when absent), bn module, conv_node and conv module. Observes min/max of the output via
    # its base.
    def __init__(self, quantizer, node):
    # [FUNCTION DOC] HELPER: Walks backward from the matched node. If its module is a ReLU,
    # records it and steps to the first argument; if that is a BatchNorm2d, records the node
    # and module and steps again. The remaining node must call a Conv2d module, otherwise
    # raises AssertionError. Stores the conv node and module. Modules are looked up in the
    # quantizer's module mapping.
        super().__init__(quantizer, node)
        self.relu_node, self.bn_node = None, None
        if isinstance(quantizer.modules[node.target], torch.nn.ReLU):
            self.relu_node = node
            node = node.args[0]
        if isinstance(quantizer.modules[node.target], torch.nn.BatchNorm2d):
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            node = node.args[0]
        if not isinstance(quantizer.modules[node.target], torch.nn.modules.Conv2d):
            raise AssertionError(
                f"Expected Conv2d, got {type(quantizer.modules[node.target])}"
            )
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]

    def quantize(self, quantizer, node, load_arg):
        # [FUNCTION DOC] HELPER: Replaces the float conv (and optional batch norm and relu)
        # by a quantized conv. Fuses batch norm into the weights if present, quantizes the
        # weights per tensor to signed 8-bit, builds the intrinsic ConvReLU2d when a relu
        # was matched or a plain quantized Conv2d otherwise, and sets its weight, bias,
        # output scale and zero point from the observed activation range. Side effects:
        # installs the quantized module in the parent module and replaces the batch norm
        # submodule with an identity module. Returns a module-call node in the new graph
        # using the conv target and the loaded first input.
        mod = self.conv
        weight, bias = mod.weight, mod.bias

        if self.bn_node is not None:
            weight, bias = fuse_conv_bn_weights(
                weight,
                bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias,
            )

        min_val, max_val = float(weight.min()), float(weight.max())

        act_scale, act_zp = self.scale_zeropoint()

        weight_scale, weight_zp = _minmax_scale_zeropoint(min_val, max_val)
        qweight = torch.quantize_per_tensor(
            weight, weight_scale, weight_zp, torch.qint8
        )

        ctor = (
            torch.ao.nn.intrinsic.quantized.ConvReLU2d
            if self.relu_node is not None
            else torch.ao.nn.quantized.Conv2d
        )

        qconv = ctor(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            mod.padding_mode,
        )

        qconv.set_weight_bias(qweight, bias)
        qconv.scale = float(act_scale)
        qconv.zero_point = int(act_zp)
        parent_name, name = _parent_name(self.conv_node.target)
        setattr(quantizer.modules[parent_name], name, qconv)
        if self.bn_node is not None:
            _, bn_name = _parent_name(self.bn_node.target)
            # we can't just delete this because submodules's forwards (which are not longer use)
            # try to call it, so replace with something that does nothing.
            setattr(quantizer.modules[parent_name], bn_name, IdentityModule())

        return quantizer.quantized_graph.create_node(
            "call_module",
            self.conv_node.target,
            (load_arg(self.conv_node.args[0]),),
            {},
        )


# turn foo.bar -> ['foo', 'bar']
def _parent_name(target):
    # [FUNCTION DOC] HELPER: Splits a dotted module path at the last dot and returns the
    # pair of parent path and attribute name; the parent is the empty string when there is
    # no dot.
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]


class DefaultQuant(MinMaxObserver):
    # [CLASS DOC] HELPER: Default constructor for input quantizers. Observes min/max like
    # its base and quantizes a node's value to unsigned 8-bit per tensor.
    def quantize(self, input):
        # [FUNCTION DOC] HELPER: Returns a graph node that quantizes the given input node to
        # quint8 per tensor using the observed scale and zero point. Raises AssertionError
        # if a non-tensor value was observed.
        # KNOWN ISSUE: the AssertionError message says the flag should be non-empty, which
        #     does not describe the actual condition (all observed values must be tensors).
        if not self.all_tensors:
            raise AssertionError("Expected self.all_tensors to be non-empty")
        scale, zeropoint = self.scale_zeropoint()
        return torch.quantize_per_tensor(
            Proxy(input), scale, zeropoint, torch.quint8
        ).node


def matches(modules, node, pattern, max_uses=sys.maxsize):
    # [FUNCTION DOC] HELPER: Checks whether a node matches a pattern. A pattern is either a
    # single matcher or a tuple of a matcher for the node followed by matchers for its
    # arguments. Returns False if the node has more users than max_uses. A module-class
    # matcher requires a module-call node whose module is an instance; another callable
    # requires a function-call node with that exact target; anything else is compared with
    # the node target. For tuples the argument count must equal the number of sub-patterns
    # and each argument must match recursively with at most one user. Returns a bool.
    if isinstance(pattern, tuple):
        self_match, *arg_matches = pattern
    else:
        self_match = pattern
        arg_matches = None

    if len(node.users) > max_uses:
        return False

    if isinstance(self_match, type) and issubclass(self_match, torch.nn.Module):
        if node.op != "call_module":
            return False
        if not isinstance(modules[node.target], self_match):
            return False
    elif callable(self_match):
        if node.op != "call_function" or node.target is not self_match:
            return False
    elif node.target != self_match:
        return False

    if not arg_matches:
        return True

    if len(arg_matches) != len(node.args):
        return False

    return all(
        matches(modules, node, arg_match, max_uses=1)
        for node, arg_match in zip(node.args, arg_matches)
    )


class Quantizer:
    # [CLASS DOC] HELPER: Experimental quantizer for a traced module. State: root module,
    # graph, quant_ctor, state_dict and modules mapping, matches (node name to a pair of
    # root node and handler), quants (input node name to observer), and quantized_graph (set
    # by quantize). Typical flow: construct, call observe with sample inputs, then call
    # quantize.
    def __init__(
        self, mod, patterns=_DEFAULT_QUANTIZATION_PATTERNS, quant_ctor=DefaultQuant
    ):
        # [FUNCTION DOC] HELPER: Stores the traced module, its graph and the constructor
        # used for input quantizers; caches the state dict and a mapping of module names to
        # modules; then computes the pattern matches and the set of input observers. The
        # default patterns argument is the shared module-level registry.
        self.root = mod
        self.graph = mod.graph
        self.quant_ctor = quant_ctor

        # cached information for observe
        self.state_dict = self.root.state_dict()
        self.modules = dict(self.root.named_modules())

        # match the patterns that will get quantized
        self.matches = self._find_matches(patterns)
        # find _inputs_ to matched nodes that are not quantized, these
        # have to be quantized, which requires measuring stats,
        # initialize an quant_ctor object for each
        self.quants = self._find_quants(quant_ctor)

    def observe(self, args):
        # [FUNCTION DOC] HELPER: Runs the graph as a simple interpreter on the given
        # positional inputs, keeping an environment of node name to value. After each node
        # it lets the matched handler (if the node is a match root) and any input observer
        # for that node record the value. On the output node returns the loaded output
        # value. Raises RuntimeError if the graph has no output node. Side effect: updates
        # the observers' statistics.
        # most of this function is just an interpreter for the graph
        # it would be possible to put this in some abstraction, but
        # it is pretty nice to just be able to see exactly what is happening here
        # and hack on it.
        # maybe we should just provide an example interpreter that people copy/paste
        # then edit.
        args_iter = iter(args)
        env = {}

        def load_arg(a):
            # [FUNCTION DOC] HELPER: Maps every node inside the given argument structure to
            # its computed value from the environment.
            return map_arg(a, lambda node: env[node.name])

        for node in self.graph.nodes:
            if node.op == "placeholder":
                result = next(args_iter)
            elif node.op == "get_attr":
                result = self.state_dict[node.target]
            elif node.op == "call_function":
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == "call_method":
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == "call_module":
                result = self.modules[node.target](
                    *load_arg(node.args), **load_arg(node.kwargs)
                )
            elif node.op == "output":
                return load_arg(node.args[0])

            env[node.name] = result
            root_node, obj = self.matches.get(node.name, (None, None))
            if root_node is node:
                obj.observe(node, env)
            if node.name in self.quants:
                self.quants[node.name].observe(node, env)

        raise RuntimeError("Graph had no output node!")

    def quantize(self):
        # [FUNCTION DOC] HELPER: Builds a new graph from the old one and returns a
        # GraphModule over the original root. Unmatched nodes are copied; match roots are
        # converted by their handler, and if the handler returns NotImplemented the root
        # node is copied instead. Matched non-root nodes are skipped. Sets quantized_graph
        # and may replace submodules of the root via the handlers.
        self.quantized_graph = Graph()

        env = {}
        quant_env = {}

        def load_arg(n, quantized):
            # [FUNCTION DOC] HELPER: Fetches a node's value in the new graph. For
            # unquantized use, a dequantize node is created from the quantized value if only
            # that exists; for quantized use, the input observer quantizes the float value
            # if only that exists. Results are cached in the respective mapping and
            # returned.
            if not quantized:
                if n.name not in env and n.name in quant_env:
                    env[n.name] = Proxy(quant_env[n.name]).dequantize().node
                return env[n.name]
            else:
                if n.name not in quant_env and n.name in env:
                    quant_env[n.name] = self.quants[n.name].quantize(env[n.name])
                return quant_env[n.name]

        def copy_recursive(node):
            # [FUNCTION DOC] HELPER: Copies one node into the new graph with its arguments
            # loaded in unquantized form, records it in the float environment and returns
            # the new node.
            r = env[node.name] = self.quantized_graph.node_copy(
                node, lambda n: load_arg(n, quantized=False)
            )
            return r

        for node in self.graph.nodes:
            root_node, obj = self.matches.get(node.name, (None, None))
            if root_node is None:
                # not quantized just copy it
                env[node.name] = self.quantized_graph.node_copy(
                    node, lambda n: load_arg(n, quantized=False)
                )

            elif root_node is node:
                r = obj.quantize(
                    self,
                    node,
                    lambda a: map_arg(a, lambda n: load_arg(n, quantized=True)),
                )
                if r is NotImplemented:
                    # quantizer choose to quantize the node take the entire match, and just copy it over
                    env[node.name] = copy_recursive(node)
                else:
                    quant_env[node.name] = r

        return GraphModule(self.root, self.quantized_graph)

    def _find_matches(self, patterns):
        # [FUNCTION DOC] HELPER: Finds pattern matches by visiting nodes in reverse order,
        # skipping nodes already matched. For each registered pattern that matches, records
        # the matched nodes under their names with a pair of the root node and a handler
        # instance created from the registered class. Returns the mapping of node name to
        # that pair.
        modules = dict(self.root.named_modules())
        match_map = {}  # node name -> (root_node, match_value?)

        def apply_match(pattern, node, match):
            # [FUNCTION DOC] HELPER: Recursively records the match for the node: for a tuple
            # pattern the first element applies to the same node and the rest to the node's
            # positional arguments; otherwise stores the match under the node name in the
            # enclosing mapping.
            if isinstance(pattern, tuple):
                s, *args = pattern
                apply_match(s, node, match)
                for subpattern, arg in zip(args, node.args):
                    apply_match(subpattern, arg, match)
            else:
                match_map[node.name] = match

        for node in reversed(self.graph.nodes):
            if node.name not in match_map:
                for pattern, value in patterns.items():
                    if matches(modules, node, pattern):
                        apply_match(pattern, node, (node, value(self, node)))

        return match_map

    def _find_quants(self, quant_ctor):
        # [FUNCTION DOC] HELPER: Creates an input observer, using the given constructor, for
        # every node that is an argument or keyword argument of a matched node. At most one
        # per distinct node name. Returns the mapping of node name to observer.
        quants = {}

        def visit_arg(n):
            # [FUNCTION DOC] HELPER: Creates and stores an observer for the node unless one
            # exists. Done even for already-quantized values, since a match may later
            # decline with NotImplemented.
            # note: we have to measure quantization information
            # even for nodes where we might not use it because it is already
            # quantized. This is because each match has the option to
            # say NotImplemented (if for instance, it is an __add__ and the data type is not appropriate)
            if n.name not in quants:
                quants[n.name] = quant_ctor(self, n)

        for node in self.graph.nodes:
            if node.name in self.matches:
                map_arg(node.args, visit_arg)
                map_arg(node.kwargs, visit_arg)
        return quants
