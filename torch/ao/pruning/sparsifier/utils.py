# mypy: allow-untyped-defs
# [MODULE DOC] Helper functions for sparsifiers: checking for a parametrization on a
# module, swapping a dense module for a sparse one, converting between modules and
# fully qualified names, splitting a tensor name into its parts, and the FakeSparsity
# mask parametrization. Exports all of these. No import-time side effects.
from itertools import chain
from typing import Any

from torch import nn
from torch.nn.utils.parametrize import is_parametrized, type_before_parametrizations


__all__ = [
    "module_contains_param",
    "swap_module",
    "module_to_fqn",
    "fqn_to_module",
    "get_arg_info_from_tensor_fqn",
    "FakeSparsity",
]


def module_contains_param(module: nn.Module, parametrization: type[nn.Module]) -> bool:
    # [FUNCTION DOC] Returns True if module is parametrized and any parametrization on any of
    # its tensors is an instance of the given class; otherwise False.
    if is_parametrized(module):
        # see if any of the module tensors have a parametrization attached that matches the one passed in
        return any(
            any(isinstance(param, parametrization) for param in param_list)
            for param_list in module.parametrizations.values()  # type: ignore[union-attr,operator]
        )
    return False


def swap_module(
    mod: nn.Module, mapping: dict[type[nn.Module], type[nn.Module]]
) -> nn.Module:
    # [FUNCTION DOC] If the type of mod before parametrization is a key in mapping, builds the
    # sparse module through the target class's from_dense, copies over forward pre-hooks
    # and forward hooks, and moves it to the original module's single device. Raises
    # AssertionError if the original has parameters or buffers on more than one device.
    # Returns the new module, or mod itself if its type is not in mapping.
    r"""Swaps the module using from_dense according to the mapping passed in.
    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to sparse nn module
    Return:
        The corresponding sparse module of `mod` according to mapping, created using from_dense
    """
    if type_before_parametrizations(mod) in mapping:
        sparse_mod = mapping[type_before_parametrizations(mod)]

        # TODO Fix this typing, as Type[Module] has no attribute "from_dense"
        new_mod = sparse_mod.from_dense(mod)  # type: ignore[attr-defined]

        # Preserve module's pre forward hooks. They'll be called on quantized input
        for pre_hook_fn in mod._forward_pre_hooks.values():
            new_mod.register_forward_pre_hook(pre_hook_fn)
        # Preserve module's post forward hooks except _observer_forward_hook
        # After convert they'll work with quantized output
        for hook_fn in mod._forward_hooks.values():
            new_mod.register_forward_hook(hook_fn)

        # respect device affinity when swapping modules
        # pyrefly: ignore [bad-argument-type]
        devices = {p.device for p in chain(mod.parameters(), mod.buffers())}
        if len(devices) > 1:
            raise AssertionError(
                f"swap_module only works with cpu or single-device CUDA modules, but got devices {devices}"
            )
        device = next(iter(devices)) if len(devices) > 0 else None
        if device:
            new_mod.to(device)

        return new_mod

    else:
        return mod


def module_to_fqn(model: nn.Module, module: nn.Module, prefix: str = "") -> str | None:
    # [FUNCTION DOC] Searches model's children recursively for module and returns its dotted
    # fully qualified name, an empty string if module is model itself, or None if it is
    # not a descendant. The prefix is used internally during recursion.
    """
    Returns the fqn for a module or None if module not a descendant of model.
    """
    if module is model:
        return ""
    for name, child in model.named_children():
        fqn = module_to_fqn(child, module, ".")
        if isinstance(fqn, str):
            return prefix + name + fqn
    return None


def fqn_to_module(model: nn.Module | None, path: str) -> nn.Module | None:
    # [FUNCTION DOC] Follows the dotted path by attribute lookup from model (so it also reaches
    # tensors) and returns the result; any missing step yields None from then on. An
    # empty path returns model.
    """
    Given an fqn, returns the corresponding module or tensor or None if the fqn given by `path`
    doesn't correspond to anything. Similar to model.get_submodule(path) but works for tensors.
    """
    if path != "":
        for name in path.split("."):
            model = getattr(model, name, None)
    return model


def get_arg_info_from_tensor_fqn(model: nn.Module, tensor_fqn: str) -> dict[str, Any]:
    # [FUNCTION DOC] Splits tensor_fqn at its last dot into module path and tensor name,
    # resolves the module (None if absent), and returns a mapping with module_fqn,
    # module, tensor_name and tensor_fqn.
    """
    Uses tensor_fqn to obtain a dict containing module_fqn, module and tensor_name
    """
    # string manip to split tensor_fqn into module_fqn and tensor_name
    # if tensor_fqn is 'weight' then module_fqn and tensor_name are '' and 'weight'
    # if tensor_fqn is 'linear.weight' then module_fqn and tensor_name are 'linear' and 'weight'
    tensor_name = tensor_fqn.rsplit(".", maxsplit=1)[-1]
    module_fqn = tensor_fqn[: -len(tensor_name) - ("." in tensor_fqn)]

    module = fqn_to_module(model, module_fqn)

    return {
        "module_fqn": module_fqn,
        "module": module,
        "tensor_name": tensor_name,
        "tensor_fqn": tensor_fqn,
    }


# Parametrizations
class FakeSparsity(nn.Module):
    # [CLASS DOC] Parametrization applying an element-wise mask, held as buffer mask, to a
    # tensor. The mask object identity must not change after it is passed in. Its
    # state_dict is empty so masks are not saved with the module.
    r"""Parametrization for the weights. Should be attached to the 'weight' or
    any other parameter that requires a mask applied to it.

    Note::

        Once the mask is passed, the variable should not change the id. The
        contents of the mask can change, but the mask reference itself should
        not.
    """

    def __init__(self, mask):
        # [FUNCTION DOC] Registers mask as the buffer mask.
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        # [FUNCTION DOC] Returns the mask multiplied element-wise with x. Raises AssertionError
        # if the mask shape differs from the shape of x.
        if self.mask.shape != x.shape:
            raise AssertionError(
                f"mask shape ({self.mask.shape}) must match x shape ({x.shape})"
            )
        return self.mask * x

    def state_dict(self, *args, **kwargs):
        # [FUNCTION DOC] Returns an empty mapping so the mask is not saved with the module.
        # We don't want to let the parametrizations to save the mask.
        # That way we make sure that the linear module doesn't store the masks
        # alongside their parametrizations.
        return {}
