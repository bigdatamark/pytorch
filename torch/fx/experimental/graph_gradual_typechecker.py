# [MODULE DOC]
# Gradual type checking and refinement over FX graphs whose nodes carry TensorType
# annotations (with Dyn as the unknown type). Contains: three module-level rule
# registries (inference, refinement and algebraic-expression rules keyed by call
# target or module class) with their register decorators; per-operation rules (add,
# getattr, transpose, reshape, batchnorm, conv, relu, maxpool, linear, adaptive
# average pool, flatten) that compute or check node types; helper functions for
# broadcasting and greatest upper bounds; the GraphTypeChecker class that applies
# inference rules to every node; and the Refine class that generates equality
# constraints and symbolic (sympy) relations. Import-time side effect: rules are
# registered into the registries, and registering a target twice raises RuntimeError.
import itertools
import operator
from collections.abc import Callable
from functools import reduce
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import sympy

import torch
from torch.fx.experimental.refinement_types import Equality
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
from torch.fx.node import Node, Target
from torch.fx.tensor_type import Dyn, is_consistent, is_more_precise, TensorType
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d


_T = TypeVar("_T")
_P = ParamSpec("_P")

_INFERENCE_RULES: dict[Target, Callable[..., Any]] = {}
_REFINEMENT_RULES: dict[Target, Callable[..., Any]] = {}
_RULES: dict[Target, Callable[..., Any]] = {}

__all__ = [
    "GraphTypeChecker",
    "Refine",
    "adaptiveavgpool2d_check",
    "adaptiveavgpool2d_inference_rule",
    "add_inference_rule",
    "all_eq",
    "bn2d_inference_rule",
    "broadcast_types",
    "calculate_out_dimension",
    "conv2d_inference_rule",
    "conv_refinement_rule",
    "conv_rule",
    "element_wise_eq",
    "expand_to_tensor_dim",
    "first_two_eq",
    "flatten_check",
    "flatten_inference_rule",
    "flatten_refinement_rule",
    "get_attr_inference_rule",
    "get_greatest_upper_bound",
    "get_parameter",
    "linear_check",
    "linear_inference_rule",
    "linear_refinement_rule",
    "maxpool2d_check",
    "maxpool2d_inference_rule",
    "register_algebraic_expressions_inference_rule",
    "register_inference_rule",
    "register_refinement_rule",
    "relu_inference_rule",
    "reshape_inference_rule",
    "transpose_inference_rule",
]


# TODO: narrow t to TensorType | _DynType once Node.type is narrowed
def expand_to_tensor_dim(t: Any, n: int) -> TensorType:
    # [FUNCTION DOC] Returns a tensor type of rank n: Dyn becomes a tensor of n Dyn
    # dimensions; a TensorType of rank n is returned unchanged. Raises TypeError for
    # a TensorType of a different rank or for any other kind of type. Pure.
    """
    Expand a type to the desired tensor dimension if possible
    Raise an error otherwise.
    - t is the given type
    - n is a number of dimensions to expand to
    """
    if t == Dyn:
        dims = [Dyn] * n
        return TensorType(tuple(dims))
    elif isinstance(t, TensorType):
        if len(t.dims) != n:
            raise TypeError(
                f"Cannot extend tensor. Tensor {t} has rank {len(t.dims)}. It should have rank {n}"
            )
        return t
    else:
        raise TypeError(f"Cannot match the type {t}")


def broadcast_types(t1: Any, t2: Any) -> tuple[Any, Any]:
    # [FUNCTION DOC] Returns the pair of types after broadcasting. If either is Dyn
    # or a unification variable, both are returned unchanged. For two tensor types,
    # the shorter one is left-padded with dimension 1 to equal rank, then each
    # dimension equal to 1 is replaced by the other tensor's dimension at that
    # position. Raises TypeError for any other combination. Inputs are not mutated.
    """
    Applies broadcasting to both given types such that they
    become consistent with each other and returns two new
    resulting types
    """

    # if either type is Dyn, do nothing since the types are already consistent
    if t1 == Dyn or t2 == Dyn or isinstance(t1, Var) or isinstance(t2, Var):
        return t1, t2

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        s1 = len(t1.dims)
        s2 = len(t2.dims)

        new_t1 = list(t1.dims)
        new_t2 = list(t2.dims)

        # We make the types the same length which is the first requirement
        # for consistency
        if s1 > s2:
            for _ in range(s1 - s2):
                new_t2.insert(0, 1)

        elif s2 > s1:
            for _ in range(s2 - s1):
                new_t1.insert(0, 1)

        # we replace occurrences of "1" with each tensor with
        # the corresponding type from the other tensor
        for i, (x, y) in enumerate(zip(new_t1, new_t2)):
            if x == 1:
                new_t1[i] = y
            elif y == 1:
                new_t2[i] = x

        # at this point our tensors should be consistent
        # and we can apply the element-wise operation and find the right dimension
        # for the output of the operation
        (t1, t2) = TensorType(tuple(new_t1)), TensorType(tuple(new_t2))
        return (t1, t2)
    else:
        raise TypeError(f"Cannot broadcast types {t1} and {t2}")


def register_inference_rule(
    call_target: Target,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # [FUNCTION DOC] Decorator factory: returns a decorator that records the decorated
    # function in the inference-rule registry under call_target and returns the
    # function unchanged.
    def register(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        # [FUNCTION DOC] HELPER: Adds fn to the inference registry; raises RuntimeError
        # if call_target is already registered. Mutates global state.
        if call_target in _INFERENCE_RULES:
            raise RuntimeError(f"Inference rule already registered for {call_target}!")
        _INFERENCE_RULES[call_target] = fn
        return fn

    return register


def register_refinement_rule(
    call_target: Target,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # [FUNCTION DOC] Decorator factory: returns a decorator that records the decorated
    # function in the refinement-rule registry under call_target and returns the
    # function unchanged.
    def register(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        # [FUNCTION DOC] HELPER: Adds fn to the refinement registry; raises
        # RuntimeError if call_target is already registered. Mutates global state.
        if call_target in _REFINEMENT_RULES:
            raise RuntimeError(f"Refinement rule already registered for {call_target}!")
        _REFINEMENT_RULES[call_target] = fn
        return fn

    return register


def register_algebraic_expressions_inference_rule(
    call_target: Target,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # [FUNCTION DOC] Decorator factory: returns a decorator that records the decorated
    # function in the algebraic-expression rule registry under call_target and
    # returns the function unchanged.
    def register(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        # [FUNCTION DOC] HELPER: Adds fn to the algebraic rule registry; raises
        # RuntimeError if call_target is already registered. Mutates global state.
        if call_target in _RULES:
            raise RuntimeError(f"Rule already registered for {call_target}!")
        _RULES[call_target] = fn
        return fn

    return register


@register_inference_rule(torch.add)
@register_inference_rule(operator.add)
def add_inference_rule(n: Node) -> Any:
    # [FUNCTION DOC] Infers the type of an addition node whose two arguments are
    # nodes (AssertionError otherwise). A scalar int plus a tensor takes the tensor's
    # type. Otherwise the operand types are broadcast; the node meta records whether
    # broadcasting changed anything and, if so, the broadcast types keyed by operand
    # name. If the (broadcast) types are consistent, the node type is set to the less
    # precise one and returned; otherwise raises TypeError. Mutates n.type and n.meta.
    """
    Apply the addition inference rule. This includes:
    - scalar addition
    - broadcasting semantics

    Note that we always return the least precise type between
    the operands (after applying broadcasting) to be the final type of the operation

    Note that we do not modify the operand types themselves after applying broadcasting
    to them. We only use them to calculate the final type
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if not isinstance(n.args[1], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[1])}")
    t1 = n.args[0].type
    t2 = n.args[1].type

    # handle scalar addition
    if t1 is int and isinstance(t2, TensorType):
        n.type = t2
        return n.type

    # handle scalar addition
    elif t2 is int and isinstance(t1, TensorType):
        n.type = t1
        return n.type

    # we bring the new types to the point where
    # we can check for consistency
    # any inconsistency would not have been caused
    # by broadcasting at this point
    (new_t1, new_t2) = broadcast_types(t1, t2)

    if new_t1 != t1 or new_t2 != t2:
        n.meta["broadcast"] = True
        n.meta[str(n.args[0])] = new_t1
        n.meta[str(n.args[1])] = new_t2

    else:
        n.meta["broadcast"] = False

    new_t1 = t1 if not n.meta["broadcast"] else new_t1
    new_t2 = t2 if not n.meta["broadcast"] else new_t2

    # we check for consistency between the new types
    if is_consistent(new_t1, new_t2):
        # we return the less precise type because
        # broadcasting may have happened
        # for operands with shape [1,2,Dyn] and [1,2,1]
        # we have to assign the node [1,2,Dyn]
        if is_more_precise(new_t1, new_t2):
            n.type = new_t2
        else:
            n.type = new_t1
        return n.type
    else:
        raise TypeError(
            f"Cannot add arguments {n.args[0]} ({n.args[0].type}) and {n.args[1]} ({n.args[1].type}) in node {n}."
            f" Types should match "
        )


@register_inference_rule(getattr)
def get_attr_inference_rule(n: Node, traced: Any) -> Any:
    # [FUNCTION DOC] Inference rule for getattr calls. If the attribute name (second
    # argument) is shape, sets the node type to Dyn and returns it; any other
    # attribute raises TypeError. The traced argument is unused. Mutates n.type.
    """
    The current getattr rule only handles the shape attribute
    Can be extended to other attributes
    The most representative type we have is "Dyn" but the system
    can be extended with more types, such as a type to represent shapes
    """
    attr_name = n.args[1]

    if attr_name == "shape":
        n.type = Dyn
    else:
        raise TypeError("Not yet implemented")

    # TODO. We leave it like this till we add a type to represent tensor sizes
    return n.type


@register_inference_rule(torch.transpose)
def transpose_inference_rule(n: Node) -> Any:
    # [FUNCTION DOC] Inference rule for transpose. Requires a node input and two int
    # dimensions (AssertionError otherwise). Dyn input gives Dyn. For a tensor type,
    # both dimensions must be non-negative and below the rank; the swapped type is
    # combined with the existing node type via greatest upper bound and stored in
    # n.type. Out-of-range dimensions or non-tensor types raise TypeError.
    # KNOWN ISSUE: negative dimensions (valid for transpose) are rejected.
    """
    We check that dimensions for the transpose operations
    are within range of the tensor type of the node
    """
    if n.target is torch.transpose:
        if not isinstance(n.args[0], Node):
            raise AssertionError(f"Expected Node, got {type(n.args[0])}")
        t = n.args[0].type

        if not isinstance(n.args[1], int):
            raise AssertionError(f"Expected int, got {type(n.args[1])}")
        if not isinstance(n.args[2], int):
            raise AssertionError(f"Expected int, got {type(n.args[2])}")
        dim1, dim2 = n.args[1], n.args[2]

        if t == Dyn:
            n.type = Dyn
            return n.type

        elif isinstance(t, TensorType):
            if 0 <= dim1 < len(t.dims) and 0 <= dim2 < len(t.dims):
                new_type = list(t.dims)
                new_type[dim1], new_type[dim2] = new_type[dim2], new_type[dim1]
                final = TensorType(new_type)
                n.type = get_greatest_upper_bound(n.type, final)
                return n.type
            else:
                raise TypeError(
                    f"Cannot transpose {dim1} and {dim2} in type {t} for node {n}"
                )
        else:
            raise TypeError(
                f"Cannot transpose {dim1} and {dim2} in type {t} for node {n}"
            )


@register_inference_rule(torch.reshape)
def reshape_inference_rule(n: Node) -> TensorType:
    # [FUNCTION DOC] Inference rule for reshape; the target shape must be a list
    # (AssertionError otherwise), with -1 mapped to Dyn. Dyn input yields the target
    # type. For a tensor input, Dyn dimensions count as 1 and the element-count
    # products must divide one another in either direction, else TypeError. Sets
    # and returns n.type.
    # KNOWN ISSUE: the product of the target list includes any -1 entry literally,
    # and the divisibility check is only a heuristic; the inner tensor-type
    # assertion is redundant.
    """
    Without dynamism, the rule checks that the
    product of the elements of the argument tensor
    type is equal to the product of the elements
    of the required shape. We gradualize this rule
    by adding a case to handle fully dynamic input
    as well as input where some of the tensor dimensions
    are unknown. In this case we check for divisibility
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    t1 = n.args[0].type

    if not isinstance(n.args[1], list):
        raise AssertionError(f"Expected list, got {type(n.args[1])}")
    t2 = n.args[1]
    t2_type = TensorType([Dyn if elem == -1 else elem for elem in t2])

    # if we do not know the original tensor dimension,
    # we return the required dimension
    if t1 == Dyn:
        n.type = t2_type
        return t2_type

    # if any of the dimensions are unknown,
    # we check for divisibility
    elif isinstance(t1, TensorType):
        if not isinstance(t1, TensorType):
            raise AssertionError(f"Expected TensorType, got {type(t1)}")
        a = [e if e != Dyn else 1 for e in t1.dims]
        p1 = reduce(operator.mul, a)
        p2 = reduce(operator.mul, t2)
        if p1 % p2 == 0 or p2 % p1 == 0:
            n.type = t2_type
            return t2_type
        else:
            raise TypeError(f"Cannot reshape in node {n} from {t1} to {t2_type}")
    else:
        raise TypeError(f"Cannot reshape in node {n} from {t1} to {t2_type}")


@register_inference_rule(BatchNorm2d)
def bn2d_inference_rule(n: Node, module_instance: Any) -> Any:
    # [FUNCTION DOC] Inference rule for BatchNorm2d. Expands the input node's type and
    # the node's own type to rank 4 (mutating the input node's type). If the channel
    # dimension of both is consistent with the module's num_features and the two
    # types are consistent, sets n.type to their greatest upper bound and returns
    # it; otherwise raises TypeError.
    """
    Given a BatchNorm2D instance and a node check the following conditions:
    - the input type can be expanded to a size 4 tensor: t =  (x_1, x_2, x_3, x_4)
    - the current node type can be expanded to a size 4 tensor: t' =  (x_1', x_2', x_3', x_4')
    - t is consistent with t'
    - x_2 is consistent with the module's num_features
    - x_2' is consistent with the module's num_features
    output type: the more precise type of t and t'
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    n.args[0].type = expand_to_tensor_dim(n.args[0].type, 4)
    arg_type = n.args[0].type
    n.type = expand_to_tensor_dim(n.type, 4)

    # we check the conditions on the incoming argument
    # and any existing annotation
    # we also check for consistency between both annotations
    if (
        is_consistent(arg_type.dims[1], module_instance.num_features)
        and is_consistent(n.type.dims[1], module_instance.num_features)
        and is_consistent(arg_type, n.type)
    ):
        # we choose the more precise type
        # to be the node type
        # so if an incoming argument has more type information
        # we set this node's type to be the argument type
        n.type = get_greatest_upper_bound(arg_type, n.type)
        return n.type
    else:
        raise TypeError(
            f"Cannot apply {module_instance} with input type {arg_type} and existing type {n.type} on {n}"
        )


def calculate_out_dimension(d_in: Any, module_instance: Any, index: int) -> Any:
    # [FUNCTION DOC] Computes one spatial output size of a conv or pool module from
    # input size d_in using the module's padding, kernel size, stride and dilation
    # (ints are expanded to pairs); index selects height (0) or width (1). Dyn
    # gives Dyn; int or sympy symbol inputs give the floor-divided formula; anything
    # else raises TypeError.
    # KNOWN ISSUE: the stride is always taken from position 0, even for width, so
    # differing strides per axis give a wrong width; the docstring mentions h_in and w_out.
    """
    For calculating h_in and w_out according to the conv2D documentation
    """
    padding = (
        (module_instance.padding, module_instance.padding)
        if isinstance(module_instance.padding, int)
        else module_instance.padding
    )
    kernel_size = (
        (module_instance.kernel_size, module_instance.kernel_size)
        if isinstance(module_instance.kernel_size, int)
        else module_instance.kernel_size
    )
    stride = (
        (module_instance.stride, module_instance.stride)
        if isinstance(module_instance.stride, int)
        else module_instance.stride
    )
    dilation = (
        (module_instance.dilation, module_instance.dilation)
        if isinstance(module_instance.dilation, int)
        else module_instance.dilation
    )

    DIMENSION_TYPES = (int, sympy.Symbol)

    if d_in == Dyn:
        return Dyn

    elif isinstance(d_in, DIMENSION_TYPES):
        n = d_in + 2 * padding[index] - dilation[index] * (kernel_size[index] - 1) - 1

        return (n // stride[0]) + 1

    else:
        raise TypeError(
            f"{d_in} in {module_instance} must be a number or Dyn. Received {type(d_in)}"
        )


# TODO: narrow params/return to TensorType | _DynType once Node.type is narrowed
def get_greatest_upper_bound(type1: Any, type2: Any) -> Any:
    # [FUNCTION DOC] Returns the more precise of two types: Dyn yields the other
    # type; two tensor types must be consistent (TypeError otherwise) and the result
    # takes the more precise dimension at each position.
    # KNOWN ISSUE: for any other combination (e.g. non-tensor, non-Dyn types) it
    # silently returns None. Zip truncates to the shorter rank.
    """
    Get the most precise type that's consistent with the given types
    """
    if type1 == Dyn:
        return type2
    elif type2 == Dyn:
        return type1
    elif isinstance(type1, TensorType) and isinstance(type2, TensorType):
        if not is_consistent(type1, type2):
            raise TypeError(f"Inconsistent types {type1}, {type2}")
        gub = [
            t1 if is_more_precise(t1, t2) else t2
            for (t1, t2) in zip(type1.dims, type2.dims)
        ]
        return TensorType(tuple(gub))


@register_inference_rule(Conv2d)
def conv2d_inference_rule(n: Node, module_instance: Any) -> Any:
    # [FUNCTION DOC] Inference rule for Conv2d. Expands the input type (mutating the
    # input node) and the node type to rank 4. If the input channel dimension is
    # consistent with in_channels, builds the output type from the input batch size,
    # out_channels and computed height and width, and sets n.type to its greatest
    # upper bound with the existing node type. Otherwise raises TypeError.
    """
    Given a Conv2D instance and a node check the following conditions:
    - the input type can be expanded to a size 4 tensor: t =  (x_1, x_2, H, W)
    - the current node type can be expanded to a size 4 tensor: t' =  (x_1', x_2', x_3', x_4')
    - x_2 is consistent with the module's in_channels
    - let o = (x_1, out_channels, H_out, W_out)
    then the output is the greatest upper bound of o and the existing node type t'.
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    n.args[0].type = expand_to_tensor_dim(n.args[0].type, 4)
    arg_type = n.args[0].type
    curr_node_type = expand_to_tensor_dim(n.type, 4)

    if is_consistent(arg_type.dims[1], module_instance.in_channels):
        w_in = arg_type.dims[3]
        h_in = arg_type.dims[2]
        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)
        new_type = TensorType(
            (arg_type.dims[0], module_instance.out_channels, h_out, w_out)
        )
        gub = get_greatest_upper_bound(new_type, curr_node_type)
        n.type = gub
        return n.type
    else:
        raise TypeError(
            f"Cannot apply {module_instance} with input type {arg_type} and existing type {n.type} on {n}"
        )


@register_inference_rule(torch.nn.ReLU)
def relu_inference_rule(n: Node, module_instance: Any) -> Any:
    # [FUNCTION DOC] Inference rule for ReLU modules: output shape equals input
    # shape. If the input is Dyn and the node has a tensor type, the input type is
    # expanded to that rank (mutating the input node). If the input is a tensor type,
    # n.type becomes the greatest upper bound of input and node types. Returns n.type.
    """
    Input and output shapes should be equal.
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.dims))

    if isinstance(n.args[0].type, TensorType):
        n.type = get_greatest_upper_bound(n.args[0].type, n.type)
    return n.type


def maxpool2d_check(typ: Any, module_instance: Any) -> TensorType:
    # [FUNCTION DOC] Returns the max pool output type for tensor type typ of rank 3
    # or 4: the last two dimensions are replaced by the computed height and width;
    # others are kept. Other ranks raise TypeError.
    """
    Applies the maxpool2d shape information to the input
    this affects the last two dimensions
    """
    new_type_list = list(typ.dims)
    if len(new_type_list) == 4 or len(new_type_list) == 3:
        w_in = new_type_list[-1]
        h_in = new_type_list[-2]

        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)

        new_type_list[-1] = w_out
        new_type_list[-2] = h_out
        return TensorType(tuple(new_type_list))

    else:
        raise TypeError(f"Wrong size {typ} for {module_instance}")


@register_inference_rule(torch.nn.MaxPool2d)
def maxpool2d_inference_rule(n: Node, module_instance: Any) -> Any:
    # [FUNCTION DOC] Inference rule for MaxPool2d. Expands a Dyn input to the node's
    # rank when the node type is a tensor; if the input is a tensor type, computes
    # the pooled type with maxpool2d_check and sets n.type to its greatest upper
    # bound with the existing type. Returns n.type. Mutates the input node type.
    """
    Given a MaxPool2D instance and a node check the following conditions:

    - Input size matches size 3 or 4
    - Current node type is consistent with the output type we will calculate
    - Input size matches output size and the last two dimensions of the output
      are w_out and h_out. The remaining dimensions are the same as the input
    - Our final result is the greatest upper bound of the output we calculate
      and the current node type.
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.dims))
    if isinstance(n.args[0].type, TensorType):
        output = maxpool2d_check(n.args[0].type, module_instance)
        n.type = get_greatest_upper_bound(output, n.type)
    return n.type


def linear_check(tensor_type: Any, module_instance: Any) -> TensorType:
    # [FUNCTION DOC] Returns the linear layer output type: same dimensions as the
    # input with the last replaced by out_features. Requires rank at least 2 and a
    # last dimension consistent with in_features; otherwise raises TypeError.
    """
    Checks that an input tensor type satisfies the conditions for linear operation
    and returns the output type based on in and out features given by module_instance
    """
    if len(tensor_type.dims) >= 2:
        if is_consistent(module_instance.in_features, tensor_type.dims[-1]):
            new_type_args = list(tensor_type.dims)
            new_type_args[-1] = module_instance.out_features
            return TensorType(tuple(new_type_args))
        else:
            raise TypeError(
                f"Inconsistent {module_instance.in_features} and {tensor_type.dims[-1]} in {module_instance}"
            )
    else:
        raise TypeError(f"Type {tensor_type} must have rank 2 or more.")


@register_inference_rule(torch.nn.Linear)
def linear_inference_rule(n: Node, module_instance: Any) -> Any:
    # [FUNCTION DOC] Inference rule for Linear. Expands a Dyn input to the node's rank
    # if the node type is a tensor; for a tensor input, computes the output type
    # with linear_check and sets n.type to its greatest upper bound with the existing
    # type. Returns n.type. Mutates the input node type.
    """
    Applies the shape information to the input then gets the greatest upper bound
    of the resulting type and the existing type
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.dims))
    if isinstance(n.args[0].type, TensorType):
        output_type = linear_check(n.args[0].type, module_instance)
        n.type = get_greatest_upper_bound(output_type, n.type)
    return n.type


def adaptiveavgpool2d_check(tensor_type: Any, module_instance: Any) -> TensorType:
    # [FUNCTION DOC] Returns the adaptive average pool output type for a tensor type
    # of rank 3 or 4 (TypeError otherwise): the last two dimensions become the
    # module's output size (an int is used for both; a tuple with a None entry
    # copies the other entry). Other dimensions are kept.
    # KNOWN ISSUE: output_size given as a list is used as is, with no None handling.
    output_size = module_instance.output_size
    if isinstance(output_size, int):
        output_size = [output_size, output_size]
    elif isinstance(output_size, tuple):
        output_size = list(output_size)
        if output_size[0] is None:
            output_size[0] = output_size[1]
        if output_size[1] is None:
            output_size[1] = output_size[0]

    new_type_list = list(tensor_type.dims)

    if len(tensor_type.dims) == 4 or len(tensor_type.dims) == 3:
        new_type_list[-1] = output_size[1]
        new_type_list[-2] = output_size[0]

        return TensorType(tuple(new_type_list))

    else:
        raise TypeError(f"Tensor ranks must be 3 or 4. Got {tensor_type}")


@register_inference_rule(torch.nn.AdaptiveAvgPool2d)
def adaptiveavgpool2d_inference_rule(n: Node, module_instance: Any) -> Any:
    # [FUNCTION DOC] Inference rule for AdaptiveAvgPool2d. Expands a Dyn input to the
    # node's rank when the node type is a tensor; for a tensor input, computes the
    # pooled type and sets n.type to its greatest upper bound with the existing
    # type. Returns n.type. Mutates the input node type.
    """
    The input and output sizes should be the same except for the last
    two dimensions taken from the input, which represent width and height
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.dims))
    if isinstance(n.args[0].type, TensorType):
        output_type = adaptiveavgpool2d_check(n.args[0].type, module_instance)
        n.type = get_greatest_upper_bound(n.type, output_type)
    return n.type


def flatten_check(tensor_type: Any, start_dim: int, end_dim: int) -> TensorType:
    # [FUNCTION DOC] Returns the flattened type: dimensions from start_dim through
    # end_dim inclusive are merged into one (their product, or Dyn if any is Dyn).
    # A negative end_dim counts from the end. Raises TypeError if the range is
    # invalid.
    # KNOWN ISSUE: start_dim of -1 is mapped to the rank (always out of range) and
    # other negative start values are replaced by their absolute value, so negative
    # start dimensions are not handled like torch.flatten.
    l = len(tensor_type.dims)

    start_dim = l if start_dim == -1 else abs(start_dim)
    end_dim = l + end_dim + 1 if end_dim < 0 else end_dim + 1

    if 0 <= start_dim <= (l - 1) and 0 <= end_dim <= l and start_dim < end_dim:
        my_args = list(tensor_type.dims)
        lhs = my_args[0:start_dim]
        rhs = my_args[end_dim:]
        mid = my_args[start_dim:end_dim]
        if Dyn in mid:
            mid = [Dyn]
        else:
            mid = [reduce(operator.mul, my_args[start_dim:end_dim])]
        new_type_list = lhs + mid + rhs
        return TensorType(tuple(new_type_list))
    else:
        raise TypeError(
            f"Incompatible dimensions {start_dim}, {end_dim - 1} in type {tensor_type}"
        )


@register_inference_rule(torch.flatten)
def flatten_inference_rule(n: Node) -> Any:
    # [FUNCTION DOC] Inference rule for flatten. Reads optional start (default 1) and
    # end (default -1) dimension arguments, which must be ints (AssertionError
    # otherwise). Expands a Dyn input to the node's rank when possible; for a tensor
    # input, computes the flattened type and sets n.type to its greatest upper bound
    # with the existing type. Returns n.type. Mutates the input node type.
    """
    Applies the flatten shape information to the input then gets the
    greatest upper bound of the resulting type and the existing type
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    # set the default start and end dims
    start_dim = 1
    end_dim = -1

    if len(n.args) > 1:
        if not isinstance(n.args[1], int):
            raise AssertionError(f"Expected int, got {type(n.args[1])}")
        start_dim = n.args[1]

    if len(n.args) > 2:
        if not isinstance(n.args[2], int):
            raise AssertionError(f"Expected int, got {type(n.args[2])}")
        end_dim = n.args[2]

    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.dims))

    if isinstance(n.args[0].type, TensorType):
        output_type = flatten_check(n.args[0].type, start_dim, end_dim)
        n.type = get_greatest_upper_bound(output_type, n.type)

    return n.type


class GraphTypeChecker:
    # [CLASS DOC] Applies the registered inference rules to every node of a traced
    # GraphModule, filling in each node's type. State: env (mapping of names to
    # values, stored but unused) and traced (the GraphModule). Key methods:
    # type_check (whole graph) and type_check_node (single node).
    def __init__(self, env: dict[str, Any], traced: torch.fx.GraphModule) -> None:
        # [FUNCTION DOC] Stores the environment mapping and the traced module.
        self.env = env
        self.traced = traced

    def type_check(self) -> bool:
        # [FUNCTION DOC] Type checks every node of the graph in order, populating node
        # types, and returns true. Failures surface as exceptions from the rules, not
        # as a false return.
        # KNOWN ISSUE: the docstring implies false is returned on failure, but the
        # method can only return true or raise.
        """
        A gradual type checker for graphs
        Effect: every node's field type will be
        populated with a type after type-checking is done
        """
        graph = self.traced.graph

        # type check every node with gradual type rules
        # if any node does not type check return false
        for n in graph.nodes:
            self.type_check_node(n)
        return True

    def type_check_node(self, n: Node) -> Any:
        # [FUNCTION DOC] Dispatches one node by its op. A missing type becomes Dyn.
        # Placeholders keep their type; get_attr nodes take the tensor shape of the
        # referenced parameter; call_function nodes use the registered rule for the
        # target (the getattr rule also receives the traced module), call_module
        # nodes use the rule for the submodule's class; output nodes take the types
        # of their arguments. Raises RuntimeError when no rule is registered and
        # NotImplementedError for other ops. Mutates n.type and returns it.
        """
        Type check a given fx node.
        Current operations:
        - Reshape
        - Transpose
        - Add
        - Relu
        - conv2d
        - batchnorm2d
        - flatten
        - maxpool2d
        - adaptiveavgpool2d
        - linear
        """
        if n.type is None:
            n.type = Dyn

        if n.op == "placeholder":
            return n.type

        elif n.op == "get_attr":
            t = get_parameter(self.traced, n.target)  # type: ignore[arg-type]
            if isinstance(t.data, torch.Tensor):
                n.type = TensorType(t.data.shape)
            return n.type

        elif n.op == "call_function":
            if n.target is getattr:
                if getattr not in _INFERENCE_RULES:
                    raise AssertionError("getattr not in _INFERENCE_RULES")
                return _INFERENCE_RULES[n.target](n, self.traced)

            elif n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](n)
            else:
                raise RuntimeError(
                    f"No inference rule registered for target {n.target}!"
                )

        elif n.op == "call_module":
            # pyrefly: ignore[bad-argument-type]
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _INFERENCE_RULES:
                return _INFERENCE_RULES[type(module_instance)](n, module_instance)
            else:
                raise RuntimeError(
                    f"No inference rule registered for class {type(module_instance)}!"
                )

        elif n.op == "output":

            def get_node_type(a: Any) -> Any:
                # [FUNCTION DOC] HELPER: Returns the type annotation of a node.
                return a.type

            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            return n.type

        else:
            raise NotImplementedError(f"Method {n.op} not yet implemented")


@register_refinement_rule(Conv2d)
def conv_refinement_rule(n: Node) -> list[Any] | None:
    # [FUNCTION DOC] Refinement rule for Conv2d: when input and node types are both
    # tensor types, returns one equality constraint between their first (batch)
    # dimensions. Otherwise returns None.
    # KNOWN ISSUE: the None result is later added to the constraint list by
    # Refine.refine_node, which fails for None.
    """
    The equality constraints are between the first dimension of
    the input and output
    """
    res: list[Any] = []
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        res = [Equality(arg_type.dims[0], n.type.dims[0])]
        return res


@register_refinement_rule(torch.nn.Linear)
def linear_refinement_rule(n: Node) -> list[Any]:
    # [FUNCTION DOC] Refinement rule for Linear: for tensor input and node types,
    # returns an equality constraint between their first dimensions; otherwise an
    # empty list.
    """
    The equality constraints are between the first dimension of
    the input and output
    """
    res: list[Any] = []
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        res = [Equality(arg_type.dims[0], n.type.dims[0])]
    return res


@register_refinement_rule(BatchNorm2d)
@register_refinement_rule(torch.nn.ReLU)
def all_eq(n: Node) -> list[Any]:
    # [FUNCTION DOC] Refinement rule for shape-preserving modules (BatchNorm2d, ReLU):
    # for tensor input and node types, returns equality constraints between every
    # pair of corresponding dimensions (length of the input); otherwise an empty list.
    """
    For operations where the input shape is equal to the output shape
    """
    res: list[Any] = []
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        args1 = arg_type.dims
        args2 = n.type.dims
        res = [Equality(args1[i], args2[i]) for i in range(len(args1))]
    return res


@register_refinement_rule(torch.nn.AdaptiveAvgPool2d)
@register_refinement_rule(torch.nn.MaxPool2d)
def first_two_eq(n: Node) -> list[Any]:
    # [FUNCTION DOC] Refinement rule for pooling modules: for tensor input and node
    # types, returns equality constraints for the first two dimensions; otherwise an
    # empty list.
    # KNOWN ISSUE: assumes rank at least 2; rank 3 or lower inputs may fail on index.
    """
    For operations where the first two dimensions of the input and output shape
    are equal
    """
    res: list[Any] = []
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        args1 = arg_type.dims
        args2 = n.type.dims
        res = [Equality(args1[0], args2[0]), Equality(args1[1], args2[1])]
    return res


@register_refinement_rule(torch.add)
@register_refinement_rule(operator.add)
def element_wise_eq(n: Node) -> list[Any]:
    # [FUNCTION DOC] Refinement rule for element-wise addition. When both arguments
    # are nodes with tensor types and the node type is a tensor, broadcasts the
    # operand types and, for each position where the two operand dimensions are
    # equal, emits an equality between that dimension and the output dimension.
    # Otherwise returns an empty list. Does not mutate types.
    """
    For element-wise operations and handles broadcasting.
    Note that after applying broadcasting to the arguments
    we are able to determine if certain dimensions have not been broadcast
    if they are symbolically equal.

    in this case, we can establish equality between those dimensions and the
    corresponding output dimensions.

    Note that it takes two iterations for this result. One iteration to establish
    equality between certain dimensions of the operands (requiring the whole solver
    including unification) and another iteration to establish equality between the operands
    and the resulting type, requiring another round of constraint generation and unification.
    """
    res: list[Any] = []
    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        arg_type1 = n.args[0].type
        arg_type2 = n.args[1].type
        if (
            isinstance(arg_type1, TensorType)
            and isinstance(arg_type2, TensorType)
            and isinstance(n.type, TensorType)
        ):
            args1, args2 = broadcast_types(arg_type1, arg_type2)
            # by this point, we know that args1 and args2 are the same size.
            a1 = args1.dims
            a2 = args2.dims
            a3 = n.type.dims

            # we would be here in the second iteration where we establish equality
            # between operand type dimensions and the resulting type dimensions
            r = []
            for x, y, z in zip(a1, a2, a3):
                if x == y:
                    r.append(Equality(x, z))
            res = r
    return res


@register_refinement_rule(torch.flatten)
def flatten_refinement_rule(n: Node) -> list[Any]:
    # [FUNCTION DOC] Refinement rule for flatten: reads optional start (default 1) and
    # end (default -1) int dimensions (AssertionError otherwise). For tensor input
    # and node types, emits equalities between the leading dimensions before the
    # start and the trailing dimensions after the end, pairing output with input
    # dimensions by position. Otherwise returns an empty list.
    # KNOWN ISSUE: trailing dimensions are paired from the same absolute index in
    # both types even though flatten changes the rank, so the pairing can be wrong.
    """
    Generates equality constraints between the dimensions of the input and output
    that will not be involved in the flatten operation
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")

    eq_const = []

    start_dim = 1
    end_dim = -1

    if len(n.args) > 1:
        if not isinstance(n.args[1], int):
            raise AssertionError(f"Expected int, got {type(n.args[1])}")
        start_dim = n.args[1]

    if len(n.args) > 2:
        if not isinstance(n.args[2], int):
            raise AssertionError(f"Expected int, got {type(n.args[2])}")
        end_dim = n.args[2]

    if isinstance(n.type, TensorType) and isinstance(n.args[0].type, TensorType):
        l = len(n.type.dims)
        arg_type = n.args[0].type
        start_dim = l if start_dim == -1 else start_dim
        end_dim = l + end_dim + 1 if end_dim < 0 else end_dim + 1

        for t1, t2 in zip(n.type.dims[0:start_dim], arg_type.dims[0:start_dim]):
            eq_const.append(Equality(t1, t2))

        for t1, t2 in zip(n.type.dims[end_dim:], arg_type.dims[end_dim:]):
            eq_const.append(Equality(t1, t2))
    return eq_const


@register_algebraic_expressions_inference_rule(Conv2d)
def conv_rule(n: Node, module_instance: Any) -> TensorType | None:
    # [FUNCTION DOC] Algebraic rule for Conv2d: for tensor input and node types,
    # keeps the node's first two dimensions and replaces height and width with
    # expressions computed from the input via calculate_out_dimension; sets and
    # returns n.type. Otherwise returns None.
    """
    Represents the output in terms of an algebraic expression w.r.t
    the input when possible
    """
    if not isinstance(n.args[0], Node):
        raise AssertionError(f"Expected Node, got {type(n.args[0])}")
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        w_in = arg_type.dims[3]
        h_in = arg_type.dims[2]
        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)
        new_type = TensorType((n.type.dims[0], n.type.dims[1], h_out, w_out))
        n.type = new_type
        return new_type


class Refine:
    # [CLASS DOC] Symbolic shape refinement over a traced module. State: constraints
    # (accumulated equality constraints), traced (the module), symbol_iter (counter
    # for fresh variable ids). refine generates constraints with fresh type
    # variables; symbolic_relations converts variables to sympy symbols and applies
    # algebraic rules.
    """
    Symbolic shape inference.
    Generates constraints over type variables.
    Currently all constraints are equality constraints.
    """

    def __init__(self, traced: Any) -> None:
        # [FUNCTION DOC] Stores the traced module, starts with an empty constraint list
        # and creates a counter starting at zero for fresh variable ids.
        self.constraints = []
        self.traced = traced
        self.symbol_iter = itertools.count(start=0, step=1)

    def refine(self) -> bool:
        # [FUNCTION DOC] Runs refine_node on every graph node in order, accumulating
        # constraints and replacing unknown types with fresh variables. Returns true.
        """
        Generates constraints for
        every node in the graph based on
        the operation.
        """
        graph = self.traced.graph
        for n in graph.nodes:
            self.refine_node(n)
        return True

    def symbolic_relations(self) -> bool:
        # [FUNCTION DOC] Runs infer_symbolic_relations on every graph node in order
        # (mutating node types). Returns true.
        """
        Infers algebraic relations
        """
        graph = self.traced.graph
        for n in graph.nodes:
            self.infer_symbolic_relations(n)
        return True

    def replace_dyn_with_fresh_var(self, typ: Any) -> Any:
        # [FUNCTION DOC] Recursively replaces each Dyn with a new unification variable
        # numbered from the counter (advancing it). Tensor types and lists are mapped
        # element-wise; other values are returned as is.
        # KNOWN ISSUE: for a tuple input it returns a lazy generator rather than a
        # tuple, and the generator consumes the counter only when iterated.
        """
        Replace all unknown types with fresh type variables.
        """
        if typ == Dyn:
            new_symbol = Var(next(self.symbol_iter))
            return new_symbol
        elif isinstance(typ, TensorType):
            new_args = [self.replace_dyn_with_fresh_var(a) for a in typ.dims]
            return TensorType(tuple(new_args))
        elif isinstance(typ, list):
            return [self.replace_dyn_with_fresh_var(t) for t in typ]
        elif isinstance(typ, tuple):
            return (self.replace_dyn_with_fresh_var(t) for t in typ)
        else:
            return typ

    def convert_to_sympy_symbols(self, typ: Any) -> Any:
        # [FUNCTION DOC] Recursively replaces each unification variable with a sympy
        # symbol named after the variable's string form. Tensor types and lists are
        # mapped element-wise; other values are returned as is.
        # KNOWN ISSUE: the docstring is copied from replace_dyn_with_fresh_var and is
        # wrong; tuple input yields a lazy generator.
        """
        Replace all unknown types with fresh type variables.
        """
        if isinstance(typ, Var):
            return sympy.symbols(str(typ))
        elif isinstance(typ, TensorType):
            new_args = [self.convert_to_sympy_symbols(a) for a in typ.dims]
            return TensorType(tuple(new_args))
        elif isinstance(typ, list):
            return [self.convert_to_sympy_symbols(t) for t in typ]
        elif isinstance(typ, tuple):
            return (self.convert_to_sympy_symbols(t) for t in typ)
        else:
            return typ

    def refine_node(self, n: Node) -> Any:
        # [FUNCTION DOC] Sets a missing type to Dyn, then replaces Dyn in the node type
        # with fresh variables. For call_function and call_module nodes with a
        # registered refinement rule (looked up by target or module class), appends
        # the rule's constraints to the constraint list. Output nodes take their
        # arguments' types and return it; other nodes return None. Mutates n.type.
        """
        Returns a list of equality constraints for
        call_module and call_function nodes.
        Models the relation between input and output dimensions
        using constraints in case they are both tensors.
        All operations used in resnet50 are defined.
        """
        if n.type is None:
            n.type = Dyn

        n.type = self.replace_dyn_with_fresh_var(n.type)

        if n.op == "call_function":
            if n.target in _REFINEMENT_RULES:
                self.constraints += _REFINEMENT_RULES[n.target](n)

        if n.op == "call_module":
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _REFINEMENT_RULES:
                self.constraints += _REFINEMENT_RULES[type(module_instance)](n)

        if n.op == "output":

            def get_node_type(a: Any) -> Any:
                # [FUNCTION DOC] HELPER: Returns the type annotation of a node.
                return a.type

            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            return n.type

    def infer_symbolic_relations(self, n: Node) -> Any:
        # [FUNCTION DOC] Converts the node's type variables to sympy symbols (mutating
        # n.type). For call_function or call_module nodes with a registered algebraic
        # rule, returns that rule's result; for output nodes sets n.type from the
        # argument types and returns it; otherwise returns None.
        n.type = self.convert_to_sympy_symbols(n.type)
        if n.op == "call_function":
            if n.target in _RULES:
                return _RULES[n.target](n)

        if n.op == "call_module":
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _RULES:
                return _RULES[type(module_instance)](n, module_instance)

        if n.op == "output":

            def get_node_type(a: Any) -> Any:
                # [FUNCTION DOC] HELPER: Returns the type annotation of a node.
                return a.type

            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            return n.type


def get_parameter(traced: Any, target: str) -> torch.nn.Parameter:
    # [FUNCTION DOC] Resolves a dotted parameter path on the traced module: the part
    # before the last dot selects a submodule (empty selects the module itself), the
    # last part is read as an attribute. Raises AttributeError if the attribute is
    # missing (the submodule lookup may also raise).
    # KNOWN ISSUE: the docstring omits the traced argument and claims an error for
    # non-Parameter values, but no type check is performed.
    """
    Returns the parameter given by ``target`` if it exists,
    otherwise throws an error.

    See the docstring for ``get_submodule`` for a more detailed
    explanation of this method's functionality as well as how to
    correctly specify ``target``.

    Args:
        target: The fully-qualified string name of the Parameter
            to look for. (See ``get_submodule`` for how to specify a
            fully-qualified string.)

    Returns:
        torch.nn.Parameter: The Parameter referenced by ``target``

    Raises:
        AttributeError: If the target string references an invalid
            path or resolves to something that is not an
            ``nn.Parameter``
    """
    module_path, _, param_name = target.rpartition(".")

    mod: torch.nn.Module = traced.get_submodule(module_path)

    if not hasattr(mod, param_name):
        raise AttributeError(mod._get_name() + " has no attribute `" + param_name + "`")

    param: torch.nn.Parameter = getattr(mod, param_name)

    return param
