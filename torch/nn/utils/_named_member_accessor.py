# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] HELPER: Utilities for reading and swapping tensors (parameters/buffers) and
# submodules of an nn.Module by dotted path: free functions set_tensor, swap_tensor and
# swap_submodule, and NamedMemberAccessor which caches submodule lookups. Used by
# functional-call style APIs. Defines a private sentinel _MISSING marking absent attributes.
from collections.abc import Iterable

import torch


_MISSING: torch.Tensor = object()  # type: ignore[assignment]


def set_tensor(module: "torch.nn.Module", name: str, tensor: torch.Tensor) -> None:
    # [FUNCTION DOC] HELPER: Sets a tensor (or None) on a module under a single-level name.
    # Replaces an existing parameter or buffer entry directly, otherwise sets a plain attribute.
    # Raises TypeError if module or tensor have the wrong type, KeyError if the name is empty or
    # contains a dot.
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{module} is not an instance of torch.nn.Module")
    if not isinstance(tensor, torch.Tensor) and tensor is not None:
        raise TypeError(f"{tensor} is not an instance of torch.Tensor")
    if "." in name:
        raise KeyError('tensor name can\'t contain "."')
    if name == "":
        raise KeyError('tensor name can\'t be empty string ""')
    if name in module._parameters:
        module._parameters[name] = tensor  # type: ignore[assignment]
    elif name in module._buffers:
        module._buffers[name] = tensor
    else:
        setattr(module, name, tensor)


def swap_tensor(
    module: "torch.nn.Module",
    name: str,
    tensor: torch.Tensor,
    allow_missing: bool = False,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Replaces a tensor attribute and returns the previous one. A
    # _MISSING value deletes the attribute. Validates types and name like set_tensor (TypeError,
    # KeyError). If the attribute does not exist, raises AttributeError unless allow_missing, in
    # which case the returned original is _MISSING. Raises TypeError if the existing value is
    # not a tensor or None.
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{module} is not an instance of torch.nn.Module")
    if (
        tensor is not _MISSING
        and not isinstance(tensor, torch.Tensor)
        and tensor is not None
    ):
        raise TypeError(f"{tensor} is not an instance of torch.Tensor")
    if "." in name:
        raise KeyError('tensor name can\'t contain "."')
    if name == "":
        raise KeyError('tensor name can\'t be empty string ""')

    orig_tensor: torch.Tensor
    if name in module._parameters:
        orig_tensor = module._parameters[name]  # type: ignore[assignment]
        if tensor is not _MISSING:
            module._parameters[name] = tensor  # type: ignore[assignment]
        else:
            del module._parameters[name]
    elif name in module._buffers:
        orig_tensor = module._buffers[name]  # type: ignore[assignment]
        if tensor is not _MISSING:
            module._buffers[name] = tensor
        else:
            del module._buffers[name]
    else:
        if hasattr(module, name):
            orig_tensor = getattr(module, name)
        else:
            if not allow_missing:
                raise AttributeError(f"{module._get_name()} has no attribute `{name}`")
            orig_tensor = _MISSING
        if (
            orig_tensor is not _MISSING
            and not isinstance(orig_tensor, torch.Tensor)
            and orig_tensor is not None
        ):
            raise TypeError(
                f"attribute `{name}`: {orig_tensor} is not an instance of torch.Tensor"
            )
        if tensor is not _MISSING:
            setattr(module, name, tensor)
        elif hasattr(module, name):
            delattr(module, name)
    # pyrefly: ignore [bad-return]
    return orig_tensor


def swap_submodule(
    module: "torch.nn.Module",
    name: str,
    submodule: "torch.nn.Module",
) -> "torch.nn.Module":
    # [FUNCTION DOC] HELPER: Replaces a direct child module and returns the old one. Raises
    # TypeError for non-module arguments or a non-module existing child, KeyError for a bad name
    # or missing child.
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{module} is not an instance of torch.nn.Module")
    if not isinstance(submodule, torch.nn.Module):
        raise TypeError(f"{submodule} is not an instance of torch.nn.Module")
    if "." in name:
        raise KeyError('submodule name can\'t contain "."')
    if name == "":
        raise KeyError('submodule name can\'t be empty string ""')
    if name not in module._modules:
        raise KeyError(f"submodule {name} does not exist")

    orig_submodule = module._modules[name]
    if not isinstance(orig_submodule, torch.nn.Module):
        raise TypeError(f"{name} attribute is not an instance of torch.nn.Module")
    module._modules[name] = submodule
    return orig_submodule


class NamedMemberAccessor:
    # [CLASS DOC] Wraps a module and offers path-based access (dotted names) to its submodules,
    # parameters and buffers, single or batched, including swapping and rollback on failure.
    # State: module and memo, a cache from path to submodule used to speed up lookups.
    # KNOWN ISSUE: the cache is never invalidated, so it can become stale after submodule swaps
    # performed through this class or directly on the module.
    """
    A class that provides a way to access the submodules and parameters/buffers of a module.

    It provides caching mechanism to speed up submodule lookups.
    This is useful for functional programming to manipulate the module state.
    """

    def __init__(self, module: "torch.nn.Module") -> None:
        # [FUNCTION DOC] Stores the module and creates an empty submodule cache.
        self.module = module
        self.memo: dict[str, torch.nn.Module] = {}

    # Nested attribute access

    def get_submodule(self, name: str) -> "torch.nn.Module":
        # [FUNCTION DOC] Returns the submodule at a dotted path (the module itself for an empty
        # path), resolving parents recursively and caching results. Raises AttributeError if a
        # component is missing and TypeError if the attribute is not a module.
        """
        Return the submodule specified by the given path.

        For example, to get the submodule mod.layer1.conv1,
        use accessor.get_submodule("layer1.conv1")

        Compare to mod.get_submodule("layer1.conv1"), this method will cache the
        intermediate submodule access to speed up future lookups.
        """
        if not name:
            return self.module

        if name in self.memo:
            return self.memo[name]
        else:
            prefix, dot, attr = name.rpartition(".")
            if dot:
                module = self.get_submodule(prefix)
            else:
                module = self.module
            try:
                submodule = getattr(module, attr)
            except AttributeError as ex:
                raise AttributeError(
                    f"{module._get_name()} has no attribute `{attr}`"
                ) from ex
            if not isinstance(submodule, torch.nn.Module):
                raise TypeError(
                    f"submodule `{name}`: {submodule} is not an instance of torch.nn.Module"
                )
            self.memo[name] = submodule
            return submodule

    def swap_submodule(self, path: str, value: "torch.nn.Module") -> "torch.nn.Module":
        # [FUNCTION DOC] Swaps the submodule at a dotted path with a new module and returns the
        # old one.
        """
        Swap the submodule specified by the given ``path`` to ``value``.

        For example, to swap the attribute mod.layer1.conv1 use
        ``accessor.swap_submodule("layer1.conv1", conv2)``.
        """
        prefix, _, attr = path.rpartition(".")
        return swap_submodule(self.get_submodule(prefix), attr, value)

    def get_tensor(self, name: str) -> torch.Tensor:
        # [FUNCTION DOC] Returns the tensor (or None) at a dotted path. Raises AttributeError if
        # absent and TypeError if the attribute is neither a tensor nor None.
        # KNOWN ISSUE: the AttributeError message uses the full path instead of the last
        # component.
        """
        Get the tensor specified by the given path to value.

        For example, to get the attribute mod.layer1.conv1.weight,
        use accessor.get_tensor('layer1.conv1.weight')

        Compare to mod.get_parameter("layer1.conv1.weight"), this method will
        cache the intermediate submodule access to speed up future lookups.
        """
        prefix, _, attr = name.rpartition(".")
        submodule = self.get_submodule(prefix)
        try:
            tensor = getattr(submodule, attr)
        except AttributeError as ex:
            raise AttributeError(
                f"{submodule._get_name()} has no attribute `{name}`"
            ) from ex
        if not isinstance(tensor, torch.Tensor) and tensor is not None:
            raise TypeError(f"{tensor} is not an instance of torch.Tensor")
        return tensor  # type: ignore[return-value]

    def set_tensor(self, name: str, value: torch.Tensor) -> None:
        # [FUNCTION DOC] Sets the tensor at a dotted path by resolving the parent and delegating
        # to set_tensor.
        """
        Set the attribute specified by the given path to value.

        For example, to set the attribute mod.layer1.conv1.weight,
        use accessor.set_tensor("layer1.conv1.weight", value)
        """
        prefix, _, attr = name.rpartition(".")
        set_tensor(self.get_submodule(prefix), attr, value)

    def del_tensor(self, name: str) -> None:
        # [FUNCTION DOC] Deletes the attribute at a dotted path. Raises AttributeError if it
        # does not exist.
        # KNOWN ISSUE: the error message uses the full path instead of the last component.
        """
        Delete the attribute specified by the given path.

        For example, to delete the attribute mod.layer1.conv1.weight,
        use accessor.del_tensor("layer1.conv1.weight")
        """
        prefix, _, attr = name.rpartition(".")
        submodule = self.get_submodule(prefix)
        try:
            delattr(submodule, attr)
        except AttributeError as ex:
            raise AttributeError(
                f"{submodule._get_name()} has no attribute `{name}`"
            ) from ex

    def swap_tensor(
        self, name: str, value: torch.Tensor, allow_missing: bool = False
    ) -> torch.Tensor:
        # [FUNCTION DOC] Swaps the tensor at a dotted path and returns the previous one
        # (_MISSING if it was absent and allow_missing is true).
        """
        Swap the attribute specified by the given path to value.

        For example, to swap the attribute mod.layer1.conv1.weight,
        use accessor.swap_tensor("layer1.conv1.weight", value)
        """
        prefix, _, attr = name.rpartition(".")
        return swap_tensor(
            self.get_submodule(prefix), attr, value, allow_missing=allow_missing
        )

    # Batched operations

    def get_tensors(self, names: Iterable[str]) -> list[torch.Tensor]:
        # [FUNCTION DOC] Returns a list with the tensor at each given path, in order.
        """
        Get the tensors specified by the given paths.

        For example, to get the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.get_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"])
        """
        return [self.get_tensor(name) for name in names]

    def set_tensors(self, names: Iterable[str], values: Iterable[torch.Tensor]) -> None:
        # [FUNCTION DOC] Sets tensors at the given paths from a parallel sequence of values.
        # Raises AssertionError if lengths differ.
        """
        Set the attributes specified by the given paths to values.

        For example, to set the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.set_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"], [weight, bias])
        """
        if not isinstance(names, (list, tuple)):
            names = list(names)
        if not isinstance(values, (list, tuple)):
            values = list(values)
        if len(names) != len(values):
            raise AssertionError(
                f"names and values must have the same length, "
                f"got {len(names)} names and {len(values)} values"
            )

        for name, value in zip(names, values, strict=True):
            self.set_tensor(name, value)

    def set_tensors_dict(self, named_tensors: dict[str, torch.Tensor]) -> None:
        # [FUNCTION DOC] Sets tensors from a mapping of dotted path to tensor.
        """
        Set the attributes specified by the given paths to values.

        For example, to set the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.set_tensors_dict({
            "layer1.conv1.weight": weight,
            "layer1.conv1.bias": bias,
        })
        """
        for name, value in named_tensors.items():
            self.set_tensor(name, value)

    def del_tensors(self, names: Iterable[str]) -> None:
        # [FUNCTION DOC] Deletes the attributes at each given path.
        """
        Delete the attributes specified by the given paths.

        For example, to delete the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.del_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"])
        """
        for name in names:
            self.del_tensor(name)

    def swap_tensors(
        self,
        names: Iterable[str],
        values: Iterable[torch.Tensor],
        allow_missing: bool = False,
    ) -> list[torch.Tensor]:
        # [FUNCTION DOC] Swaps tensors at the given paths with parallel values and returns the
        # list of previous tensors. Raises AssertionError if lengths differ. No rollback is done
        # on failure.
        """
        Swap the attributes specified by the given paths to values.

        For example, to swap the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.swap_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"], [weight, bias])
        """
        if not isinstance(names, (list, tuple)):
            names = list(names)
        if not isinstance(values, (list, tuple)):
            values = list(values)
        if len(names) != len(values):
            raise AssertionError(
                f"names and values must have the same length, "
                f"got {len(names)} names and {len(values)} values"
            )

        return [
            self.swap_tensor(name, value, allow_missing=allow_missing)
            for name, value in zip(names, values, strict=True)
        ]

    def swap_tensors_dict(
        self, named_tensors: dict[str, torch.Tensor], allow_missing: bool = False
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        # [FUNCTION DOC] Swaps tensors from a mapping, returning the mapping of originals and
        # the list of paths that were missing. If any swap raises, or a path is missing while
        # allow_missing is false, all completed swaps are undone and the error (RuntimeError for
        # missing keys) is raised.
        """
        Swap the attributes specified by the given paths to values.

        For example, to swap the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.swap_tensors_dict({
            "layer1.conv1.weight": weight,
            "layer1.conv1.bias": bias,
        })
        """
        orig_named_tensors = {}
        missing_keys = []
        try:
            for name, tensor in named_tensors.items():
                orig_tensor = self.swap_tensor(name, tensor, allow_missing=True)
                if orig_tensor is _MISSING:
                    missing_keys.append(name)
                orig_named_tensors[name] = orig_tensor
        except Exception:
            # Swap back if any exception occurs
            for name, orig_tensor in orig_named_tensors.items():
                self.swap_tensor(name, orig_tensor, allow_missing=True)
            raise
        if missing_keys and not allow_missing:
            # Swap back if any key is missing when allow_missing is False
            for name, orig_tensor in orig_named_tensors.items():
                self.swap_tensor(name, orig_tensor, allow_missing=True)
            raise RuntimeError(f"Missing key(s): {', '.join(map(repr, missing_keys))}.")
        return orig_named_tensors, missing_keys

    def check_keys(self, keys: Iterable[str]) -> tuple[list[str], list[str]]:
        # [FUNCTION DOC] Compares the given keys with all parameter and buffer names (duplicates
        # included) and returns the sorted missing keys and the sorted unexpected keys.
        """Check that the given keys are valid."""
        keys = set(keys)
        valid_keys = {name for name, _ in self.named_tensors(remove_duplicate=False)}
        missing_keys = valid_keys - keys
        unexpected_keys = keys - valid_keys
        return sorted(missing_keys), sorted(unexpected_keys)

    # Shortcut methods

    def named_parameters(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        # [FUNCTION DOC] Generator yielding the module's named parameters.
        """Iterate over all the parameters in the module."""
        yield from self.module.named_parameters(remove_duplicate=remove_duplicate)

    def named_buffers(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        # [FUNCTION DOC] Generator yielding the module's named buffers.
        """Iterate over all the buffers in the module."""
        yield from self.module.named_buffers(remove_duplicate=remove_duplicate)

    def named_tensors(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        # [FUNCTION DOC] Generator yielding all named parameters followed by all named buffers.
        """Iterate over all the tensors in the module."""
        yield from self.module.named_parameters(remove_duplicate=remove_duplicate)
        yield from self.module.named_buffers(remove_duplicate=remove_duplicate)

    def named_modules(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, "torch.nn.Module"]]:
        # [FUNCTION DOC] Generator yielding the module's named submodules.
        """Iterate over all the modules in the module."""
        yield from self.module.named_modules(remove_duplicate=remove_duplicate)
