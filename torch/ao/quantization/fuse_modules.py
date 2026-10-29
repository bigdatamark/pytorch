# mypy: allow-untyped-defs
# [MODULE DOC] Eager mode module fusion API: fuse_modules and fuse_modules_qat fuse named submodules (conv/bn/relu, linear/bn/relu, etc.)
# of a model into fused modules, replacing the remaining ones with identity modules. Also provides fuse_known_modules (default fuser function)
# and helpers to get/set nested submodules by dotted name. Re-exports fuse_conv_bn, fuse_conv_bn_relu and get_fuser_method for backward compatibility.
import copy

import torch.nn as nn

# for backward compatibility
from torch.ao.quantization.fuser_method_mappings import (  # noqa: F401  # noqa: F401
    fuse_conv_bn,
    fuse_conv_bn_relu,
    get_fuser_method,
)
from torch.nn.utils.parametrize import type_before_parametrizations


__all__ = [
    "fuse_known_modules",
    "fuse_modules",
    "fuse_modules_qat",
]


# Generalization of getattr
def _get_module(model, submodule_key):
    # [FUNCTION DOC] HELPER: Return the nested submodule of model addressed by a dotted attribute path, by repeated attribute lookup (AttributeError if missing).
    tokens = submodule_key.split(".")
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


# Generalization of setattr
def _set_module(model, submodule_key, module):
    # [FUNCTION DOC] HELPER: Set the nested submodule addressed by a dotted path to the given module, by looking up the parent through the path prefix and assigning the last name. Mutates the model.
    tokens = submodule_key.split(".")
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)

    setattr(cur_mod, tokens[-1], module)


def fuse_known_modules(mod_list, is_qat, additional_fuser_method_mapping=None):
    # [FUNCTION DOC] Fuse a list of modules into a list of the same length. Looks up the fuser method for the tuple of module types (honouring parametrizations)
    # using the default mapping plus an optional additional mapping, and calls it with the is_qat flag and the modules. Raises NotImplementedError if no fuser method is found.
    # Forward pre hooks of the first module and forward hooks of the last module are moved to the fused module and cleared on the originals (other hooks are lost).
    # Returns a list whose first element is the fused module and the rest are Identity modules copying the first module's training mode.
    # KNOWN ISSUE: get_fuser_method raises AssertionError when nothing is found, so the NotImplementedError branch is effectively unreachable.
    r"""Return a list of known fuse modules.

    Returns a list of modules that fuses the operations specified
     in the input module list.

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, bn
    linear, relu
    For these sequences, the first element in the output module list performs
    the fused operation. The rest of the elements are set to nn.Identity()
    """
    types = tuple(type_before_parametrizations(m) for m in mod_list)
    fuser_method = get_fuser_method(types, additional_fuser_method_mapping)
    if fuser_method is None:
        raise NotImplementedError(f"Cannot fuse modules: {types}")
    new_mod: list[nn.Module | None] = [None] * len(mod_list)
    fused = fuser_method(is_qat, *mod_list)
    # NOTE: forward hooks not processed in the two following for loops will be lost after the fusion
    # Move pre forward hooks of the base module to resulting fused module
    for pre_hook_fn in mod_list[0]._forward_pre_hooks.values():
        fused.register_forward_pre_hook(pre_hook_fn)
    mod_list[0]._forward_pre_hooks.clear()
    # Move post forward hooks of the last module to resulting fused module
    for hook_fn in mod_list[-1]._forward_hooks.values():
        fused.register_forward_hook(hook_fn)
    mod_list[-1]._forward_hooks.clear()
    new_mod[0] = fused

    for i in range(1, len(mod_list)):
        identity = nn.Identity()
        identity.training = mod_list[0].training
        new_mod[i] = identity

    return new_mod


def _fuse_modules_helper(
    model,
    modules_to_fuse,
    is_qat,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    # [FUNCTION DOC] HELPER: Fuse one list of module names in place. Reads the optional additional fuser method mapping from the custom config (default empty),
    # resolves the modules, calls fuser_func with (modules, is_qat, additional mapping) and writes the returned modules back at the same names. Mutates the model.
    if fuse_custom_config_dict is None:
        fuse_custom_config_dict = {}
    additional_fuser_method_mapping = fuse_custom_config_dict.get(
        "additional_fuser_method_mapping", {}
    )
    mod_list = [_get_module(model, item) for item in modules_to_fuse]

    # Fuse list of modules
    new_mod_list = fuser_func(mod_list, is_qat, additional_fuser_method_mapping)

    # Replace original module list with fused module list
    for i, item in enumerate(modules_to_fuse):
        _set_module(model, item, new_mod_list[i])


def _fuse_modules(
    model,
    modules_to_fuse,
    is_qat,
    inplace=False,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    # [FUNCTION DOC] HELPER: Shared implementation of fuse_modules and fuse_modules_qat. Deep-copies the model unless inplace. If modules_to_fuse is a flat list of strings
    # it is treated as one group, otherwise as a list of groups, each fused through the helper. Returns the model (copy or original).
    if not inplace:
        model = copy.deepcopy(model)

    if all(isinstance(module_element, str) for module_element in modules_to_fuse):
        # Handle case of modules_to_fuse being a list
        _fuse_modules_helper(
            model, modules_to_fuse, is_qat, fuser_func, fuse_custom_config_dict
        )
    else:
        # Handle case of modules_to_fuse being a list of lists
        for module_list in modules_to_fuse:
            _fuse_modules_helper(
                model, module_list, is_qat, fuser_func, fuse_custom_config_dict
            )
    return model


def fuse_modules(
    model,
    modules_to_fuse,
    inplace=False,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    # [FUNCTION DOC] Public eval-mode entry point: fuse the given module name list(s) with is_qat disabled. Args: model, modules_to_fuse (list of names or list of lists),
    # inplace flag (default copy), fuser_func, fuse_custom_config_dict. Returns the fused model.
    # KNOWN ISSUE: the docstring says a new copy is created if inplace is true; the code copies when inplace is false. Its list of supported sequences also differs from fuse_known_modules.
    r"""Fuse a list of modules into a single module.

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, relu
    bn, relu
    All other sequences are left unchanged.
    For these sequences, replaces the first item in the list
    with the fused module, replacing the rest of the modules
    with identity.

    Args:
        model: Model containing the modules to be fused
        modules_to_fuse: list of list of module names to fuse. Can also be a list
                         of strings if there is only a single list of modules to fuse.
        inplace: bool specifying if fusion happens in place on the model, by default
                 a new model is returned
        fuser_func: Function that takes in a list of modules and outputs a list of fused modules
                    of the same length. For example,
                    fuser_func([convModule, BNModule]) returns the list [ConvBNModule, nn.Identity()]
                    Defaults to torch.ao.quantization.fuse_known_modules
        `fuse_custom_config_dict`: custom configuration for fusion

    .. code-block:: python

       # Example of fuse_custom_config_dict
       fuse_custom_config_dict = {
           # Additional fuser_method mapping
           "additional_fuser_method_mapping": {
               (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn
           },
       }

    Returns:
        model with fused modules. A new copy is created if inplace=True.

    Examples::

            >>> # xdoctest: +SKIP
            >>> m = M().eval()
            >>> # m is a module containing the sub-modules below
            >>> modules_to_fuse = [ ['conv1', 'bn1', 'relu1'], ['submodule.conv', 'submodule.relu']]
            >>> fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

            >>> m = M().eval()
            >>> # Alternately provide a single list of modules to fuse
            >>> modules_to_fuse = ['conv1', 'bn1', 'relu1']
            >>> fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

    """
    return _fuse_modules(
        model,
        modules_to_fuse,
        is_qat=False,
        inplace=inplace,
        fuser_func=fuser_func,
        fuse_custom_config_dict=fuse_custom_config_dict,
    )


def fuse_modules_qat(
    model,
    modules_to_fuse,
    inplace=False,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    # [FUNCTION DOC] Same as fuse_modules but with is_qat enabled, so the training-time (QAT) fused modules are produced. Returns the fused model.
    """QAT version for `fuse_modules`."""
    return _fuse_modules(
        model,
        modules_to_fuse,
        is_qat=True,
        inplace=inplace,
        fuser_func=fuser_func,
        fuse_custom_config_dict=fuse_custom_config_dict,
    )
