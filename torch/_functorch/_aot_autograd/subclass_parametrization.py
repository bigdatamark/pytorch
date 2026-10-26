# [MODULE DOC] Utility that replaces tensor-subclass parameters/buffers of an nn.Module
# with parametrizations holding their plain inner tensors, so the subclass is rebuilt on
# access instead of being flattened on every call. Contents: SubclassCreationMeta (a small
# local record of how to rebuild a subclass), UnwrapTensorSubclass (the parametrization
# module) and unwrap_tensor_subclass_parameters (the module-tree transformation).
# No import-time side effects beyond imports.
from __future__ import annotations

import dataclasses
import itertools
from typing import Any, TYPE_CHECKING

import torch
from torch._custom_class_base import CustomClassBase
from torch._library.opaque_object import is_opaque_symbolic_type
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .schemas import OpaqueMeta


if TYPE_CHECKING:
    from collections.abc import Iterable


# This is technically very similar to SubclassCreatingMeta
# in aot_autograd, but we don't need all the stuff in there
# so just recreated a new dataclass.
@dataclasses.dataclass
class SubclassCreationMeta:
    # [CLASS DOC] Dataclass describing how to rebuild one tensor subclass from a flat list
    # of plain tensors/opaques: start_idx and num_tensors locate its slice in the flat list,
    # class_type is the subclass type, attrs maps each inner attr name to a nested meta,
    # an OpaqueMeta, or None for a plain tensor, metadata is the __tensor_flatten__ context,
    # and outer_size/outer_stride are the wrapper's size and stride. Distinct from the
    # SubclassCreationMeta in schemas (this one is simpler and local to this module).
    start_idx: int
    num_tensors: int
    class_type: Any
    # None means the attr is a plain tensor (base case of recursion)
    attrs: dict[str, SubclassCreationMeta | OpaqueMeta | None]
    metadata: Any
    outer_size: Iterable[None | int | torch.SymInt]
    outer_stride: Iterable[None | int | torch.SymInt]


class UnwrapTensorSubclass(torch.nn.Module):
    # [CLASS DOC] Parametrization module for torch.nn.utils.parametrize. right_inverse
    # flattens a subclass tensor into a list of plain tensors/opaques and records the
    # rebuild recipe in the attribute subclass_meta (set dynamically, not in __init__);
    # forward takes those flat tensors and rebuilds the original subclass via
    # __tensor_unflatten__. forward must only be used after right_inverse has set
    # subclass_meta.
    def forward(self, *tensors) -> torch.Tensor:  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Rebuilds the subclass tensor from the flat inputs using the stored
        # subclass_meta, consuming tensors in order (depth-first over attrs) and calling
        # __tensor_unflatten__ at each level. Returns the reconstructed tensor. Raises
        # AttributeError if right_inverse has not run (subclass_meta missing).
        todo: list[torch.Tensor | CustomClassBase] = list(tensors)

        def _unwrap_tensor_subclasses(subclass_meta, tensors, offset):  # type: ignore[no-untyped-def]
            # [FUNCTION DOC] HELPER: Recursive rebuild. A None meta takes the next flat
            # tensor; otherwise gathers each attr (opaque takes the next item, others
            # recurse), then calls the class's __tensor_unflatten__ with the stored
            # metadata, outer size and stride. Returns (rebuilt tensor, new offset).
            if subclass_meta is None:
                return tensors[offset], offset + 1
            inner_tensors = {}
            for attr, meta in subclass_meta.attrs.items():
                if isinstance(meta, OpaqueMeta):
                    inner_tensors[attr] = tensors[offset]
                    offset += 1
                else:
                    built_tensor, offset = _unwrap_tensor_subclasses(
                        meta, tensors, offset
                    )
                    inner_tensors[attr] = built_tensor
            rebuilt = subclass_meta.class_type.__tensor_unflatten__(
                inner_tensors,
                subclass_meta.metadata,
                subclass_meta.outer_size,
                subclass_meta.outer_stride,
            )
            return rebuilt, offset

        return _unwrap_tensor_subclasses(self.subclass_meta, todo, 0)[0]

    def right_inverse(
        self, tensor: torch.Tensor
    ) -> list[torch.Tensor | CustomClassBase]:
        # [FUNCTION DOC] Flattens a subclass tensor into a list of plain tensors and
        # reference-type opaques (depth-first over __tensor_flatten__ attrs), stores the
        # rebuild recipe in self.subclass_meta, and returns the flat list. Raises
        # AssertionError if the input is a plain torch.Tensor or an attr is neither a Tensor
        # nor an opaque; raises ValueError for a non-symbolic-type opaque attr.
        if type(tensor) is torch.Tensor:
            raise AssertionError("tensor must be a subclass, not torch.Tensor")
        plain_tensors: list[torch.Tensor | CustomClassBase] = []

        def _create_subclass_meta(tensor, idx, plain_tensor_container):  # type: ignore[no-untyped-def]
            # [FUNCTION DOC] HELPER: Recursive flatten. A plain tensor is appended to the
            # container and yields meta None. For a subclass, walks its flatten attrs,
            # building nested metas (OpaqueMeta for opaques) and appending leaves to the
            # container, and returns (SubclassCreationMeta, next index).
            if type(tensor) is torch.Tensor:
                plain_tensor_container.append(tensor)
                return None, idx + 1
            inner_tensors_attrnames, metadata = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
            new_idx = idx
            attr_to_meta: dict[str, SubclassCreationMeta | OpaqueMeta | None] = {}
            for attr in inner_tensors_attrnames:
                val = getattr(tensor, attr)
                match val:
                    case CustomClassBase():
                        if not is_opaque_symbolic_type(type(val)):
                            raise ValueError(
                                f"{type(val).__name__!r} found in tensor attrs of "
                                f"{type(tensor).__name__}.__tensor_flatten__(). "
                                "Only tensors and reference-type opaques are allowed "
                                "in tensor attrs."
                            )
                        attr_to_meta[attr] = OpaqueMeta()
                        plain_tensor_container.append(val)
                        new_idx += 1
                    case torch.Tensor():
                        subclass_meta, new_idx = _create_subclass_meta(
                            val, new_idx, plain_tensor_container
                        )
                        attr_to_meta[attr] = subclass_meta
                    case _:
                        raise AssertionError(
                            f"expected Tensor or CustomClassBase, got {type(val)}"
                        )
            return (
                SubclassCreationMeta(
                    start_idx=idx,
                    num_tensors=new_idx - idx,
                    class_type=type(tensor),
                    attrs=attr_to_meta,
                    metadata=metadata,
                    outer_size=tensor.size(),
                    outer_stride=tensor.stride(),
                ),
                new_idx,
            )

        self.subclass_meta = _create_subclass_meta(tensor, 0, plain_tensors)[0]
        return plain_tensors


def unwrap_tensor_subclass_parameters(module: torch.nn.Module) -> torch.nn.Module:
    # [FUNCTION DOC] Walks the module tree (recursing into children after handling the
    # module's own direct parameters and buffers) and registers an UnwrapTensorSubclass
    # parametrization on each parameter/buffer that is a traceable wrapper subclass.
    # Mutates the module in place (state_dict keys change) and returns the same module.
    """
    Model transformation that replaces all the parameters that are subclasses to plain tensors.
    This reduces runtime overhead of flattening/unflattening the parameters.

    This transformation adds parametrization with `torch.nn.utils.parametrize`.
    The FQNs of the subclass parameters will be changed and state_dict will become incompatible with the original model.
    E.g.
    Original model state_dict: {"p1": torch.testing._internal.TwoTensor}
    becomes: {"parametrizations.p2.original0": torch.Tensor, "parametrizations.p2.original1": torch.Tensor}

    """
    for name, tensor in itertools.chain(
        list(module.named_parameters(recurse=False)),
        # pyrefly: ignore [bad-argument-type, no-matching-overload]
        list(module.named_buffers(recurse=False)),
    ):
        if is_traceable_wrapper_subclass(tensor):
            torch.nn.utils.parametrize.register_parametrization(
                module, name, UnwrapTensorSubclass()
            )

    for child in module.children():
        unwrap_tensor_subclass_parameters(child)

    return module
