# mypy: allow-untyped-defs
# [MODULE DOC] Cross-layer equalization (data-free quantization, section 4.1 of the
# referenced paper): rescales the weights of adjacent layers so that per-channel ranges
# match, repeating until the weights converge. Contains weight/bias accessors that work
# for plain and fused modules, per-axis range helpers, pair-list preprocessing, equalize
# and converged. Defines the supported module type sets at import.
import copy
from itertools import chain
from typing import Any

import torch


__all__ = [
    "set_module_weight",
    "set_module_bias",
    "has_bias",
    "get_module_weight",
    "get_module_bias",
    "max_over_ndim",
    "min_over_ndim",
    "channel_range",
    "get_name_by_module",
    "cross_layer_equalization",
    "process_paired_modules_list_to_name",
    "expand_groups_in_paired_modules_list",
    "equalize",
    "converged",
]

_supported_types = {torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d}
_supported_intrinsic_types = {
    torch.ao.nn.intrinsic.ConvReLU2d,
    torch.ao.nn.intrinsic.LinearReLU,
    torch.ao.nn.intrinsic.ConvReLU1d,
}
_all_supported_types = _supported_types.union(_supported_intrinsic_types)


def set_module_weight(module, weight) -> None:
    # [FUNCTION DOC] Replaces the weight with a new Parameter built from weight; for fused
    # modules (type not in the plain supported set) the first submodule is changed.
    if type(module) in _supported_types:
        module.weight = torch.nn.Parameter(weight)
    else:
        module[0].weight = torch.nn.Parameter(weight)


def set_module_bias(module, bias) -> None:
    # [FUNCTION DOC] Replaces the bias with a new Parameter built from bias; for fused modules
    # the first submodule is changed.
    if type(module) in _supported_types:
        module.bias = torch.nn.Parameter(bias)
    else:
        module[0].bias = torch.nn.Parameter(bias)


def has_bias(module) -> bool:
    # [FUNCTION DOC] Returns whether the module (or the first submodule of a fused module) has a
    # bias that is not None.
    if type(module) in _supported_types:
        return module.bias is not None
    else:
        return module[0].bias is not None


def get_module_weight(module):
    # [FUNCTION DOC] Returns the weight of the module, or of its first submodule for fused modules.
    if type(module) in _supported_types:
        return module.weight
    else:
        return module[0].weight


def get_module_bias(module):
    # [FUNCTION DOC] Returns the bias of the module, or of its first submodule for fused modules.
    if type(module) in _supported_types:
        return module.bias
    else:
        return module[0].bias


def max_over_ndim(input, axis_list, keepdim=False):
    # [FUNCTION DOC] Reduces input by maximum over each listed axis in turn, highest axis
    # first, and returns the result. Sorts axis_list in place (side effect on the caller's list).
    """Apply 'torch.max' over the given axes."""
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.max(axis, keepdim)
    return input


def min_over_ndim(input, axis_list, keepdim=False):
    # [FUNCTION DOC] Reduces input by minimum over each listed axis in turn, highest axis
    # first, and returns the result. Sorts axis_list in place (side effect on the caller's list).
    """Apply 'torch.min' over the given axes."""
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.min(axis, keepdim)
    return input


def channel_range(input, axis=0):
    # [FUNCTION DOC] Returns, for each index along axis, the maximum minus the minimum of the
    # input over all other dimensions. Raises AssertionError if the result length differs
    # from the axis size.
    """Find the range of weights associated with a specific channel."""
    size_of_tensor_dim = input.ndim
    axis_list = list(range(size_of_tensor_dim))
    axis_list.remove(axis)

    mins = min_over_ndim(input, axis_list)
    maxs = max_over_ndim(input, axis_list)

    if mins.size(0) != input.size(axis):
        raise AssertionError(
            "Dimensions of resultant channel range does not match size of requested axis"
        )
    return maxs - mins


def get_name_by_module(model, module):
    # [FUNCTION DOC] Returns the dotted name of module inside model, matched by identity; raises
    # ValueError if module is not part of model.
    """Get the name of a module within a model.

    Args:
        model: a model (nn.module) that equalization is to be applied on
        module: a module within the model

    Returns:
        name: the name of the module within the model
    """
    for name, m in model.named_modules():
        if m is module:
            return name
    raise ValueError("module is not in the model")


def cross_layer_equalization(module1, module2, output_axis=0, input_axis=1):
    # [FUNCTION DOC] Rescales two adjacent layers in place. Each scaling factor is the square
    # root of the output-channel range of module1's weight over the input-channel range of
    # module2's weight (a tiny constant is added to the latter, mutating it). module1's
    # weight and bias are divided by it and module2's weight multiplied, so the
    # network function is preserved for piecewise linear activations. Raises ValueError
    # for unsupported module types and TypeError if module1's output channels differ from
    # module2's input channels.
    """Scale the range of Tensor1.output to equal Tensor2.input.

    Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    """
    if (
        type(module1) not in _all_supported_types
        or type(module2) not in _all_supported_types
    ):
        raise ValueError(
            "module type not supported:", type(module1), " ", type(module2)
        )

    bias = get_module_bias(module1) if has_bias(module1) else None

    weight1 = get_module_weight(module1)
    weight2 = get_module_weight(module2)

    if weight1.size(output_axis) != weight2.size(input_axis):
        raise TypeError(
            "Number of output channels of first arg do not match \
        number input channels of second arg"
        )

    weight1_range = channel_range(weight1, output_axis)
    weight2_range = channel_range(weight2, input_axis)

    # producing scaling factors to applied
    weight2_range += 1e-9
    scaling_factors = torch.sqrt(weight1_range / weight2_range)
    inverse_scaling_factors = torch.reciprocal(scaling_factors)

    if bias is not None:
        bias = bias * inverse_scaling_factors

    # formatting the scaling (1D) tensors to be applied on the given argument tensors
    # pads axis to (1D) tensors to then be broadcasted
    size1 = [1] * weight1.ndim
    size1[output_axis] = weight1.size(output_axis)
    size2 = [1] * weight2.ndim
    size2[input_axis] = weight2.size(input_axis)

    scaling_factors = torch.reshape(scaling_factors, size2)
    inverse_scaling_factors = torch.reshape(inverse_scaling_factors, size1)

    weight1 = weight1 * inverse_scaling_factors
    weight2 = weight2 * scaling_factors

    set_module_weight(module1, weight1)
    if bias is not None:
        set_module_bias(module1, bias)
    set_module_weight(module2, weight2)


def process_paired_modules_list_to_name(model, paired_modules_list):
    # [FUNCTION DOC] Replaces, inside each group of the list, every module with its name in
    # model (strings are kept) and returns the same list object. Raises TypeError for
    # items that are neither modules nor strings. Mutates the input groups.
    """Processes a list of paired modules to a list of names of paired modules."""

    for group in paired_modules_list:
        for i, item in enumerate(group):
            if isinstance(item, torch.nn.Module):
                group[i] = get_name_by_module(model, item)
            elif not isinstance(item, str):
                raise TypeError("item must be a nn.Module or a string")
    return paired_modules_list


def expand_groups_in_paired_modules_list(paired_modules_list):
    # [FUNCTION DOC] Returns a new list in which groups of two are kept and longer groups are
    # split into overlapping consecutive pairs. Raises ValueError for a group of one.
    # Empty groups are silently dropped.
    """Expands module pair groups larger than two into groups of two modules."""
    new_list = []

    for group in paired_modules_list:
        if len(group) == 1:
            raise ValueError("Group must have at least two modules")
        elif len(group) == 2:
            new_list.append(group)
        elif len(group) > 2:
            new_list.extend([group[i], group[i + 1]] for i in range(len(group) - 1))

    return new_list


def equalize(model, paired_modules_list, threshold=1e-4, inplace=True):
    # [FUNCTION DOC] Applies cross-layer equalization over the given pairs (modules or names;
    # groups longer than two are expanded into consecutive pairs) repeatedly until the
    # weights change by less than threshold between rounds. Works on a deep copy unless
    # inplace is true. Returns the model. Note that the names in the input list are
    # rewritten in place as a side effect.
    """Equalize modules until convergence is achieved.

    Given a list of adjacent modules within a model, equalization will
    be applied between each pair, this will repeated until convergence is achieved

    Keeps a copy of the changing modules from the previous iteration, if the copies
    are not that different than the current modules (determined by converged_test),
    then the modules have converged enough that further equalizing is not necessary

    Reference is section 4.1 of this paper https://arxiv.org/pdf/1906.04721.pdf

    Args:
        model: a model (nn.Module) that equalization is to be applied on
            paired_modules_list (List(List[nn.module || str])): a list of lists
            where each sublist is a pair of two submodules found in the model,
            for each pair the two modules have to be adjacent in the model,
            with only piece-wise-linear functions like a (P)ReLU or LeakyReLU in between
            to get expected results.
            The list can contain either modules, or names of modules in the model.
            If you pass multiple modules in the same list, they will all be equalized together.
            threshold (float): a number used by the converged function to determine what degree
            of similarity between models is necessary for them to be called equivalent
        inplace (bool): determines if function is inplace or not
    """

    paired_modules_list = process_paired_modules_list_to_name(
        model, paired_modules_list
    )

    if not inplace:
        model = copy.deepcopy(model)

    paired_modules_list = expand_groups_in_paired_modules_list(paired_modules_list)

    name_to_module: dict[str, torch.nn.Module] = {}
    previous_name_to_module: dict[str, Any] = {}
    name_set = set(chain.from_iterable(paired_modules_list))

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
            previous_name_to_module[name] = None
    while not converged(name_to_module, previous_name_to_module, threshold):
        for pair in paired_modules_list:
            previous_name_to_module[pair[0]] = copy.deepcopy(name_to_module[pair[0]])
            previous_name_to_module[pair[1]] = copy.deepcopy(name_to_module[pair[1]])

            cross_layer_equalization(name_to_module[pair[0]], name_to_module[pair[1]])

    return model


def converged(curr_modules, prev_modules, threshold=1e-4):
    # [FUNCTION DOC] Returns True if the summed vector norm of weight differences between
    # matching modules of the two name-to-module mappings is below threshold. Raises
    # ValueError if the key sets differ; returns False when any previous module is None
    # (first round).
    """Test whether modules are converged to a specified threshold.

    Tests for the summed norm of the differences between each set of modules
    being less than the given threshold

    Takes two dictionaries mapping names to modules, the set of names for each dictionary
    should be the same, looping over the set of names, for each name take the difference
    between the associated modules in each dictionary

    """
    if curr_modules.keys() != prev_modules.keys():
        raise ValueError(
            "The keys to the given mappings must have the same set of names of modules"
        )

    summed_norms = torch.tensor(0.0)
    if None in prev_modules.values():
        return False
    for name in curr_modules:
        curr_weight = get_module_weight(curr_modules[name])
        prev_weight = get_module_weight(prev_modules[name])

        difference = curr_weight.sub(prev_weight)
        summed_norms += torch.linalg.vector_norm(difference)
    return bool(summed_norms < threshold)
