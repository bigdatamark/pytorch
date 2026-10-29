# [MODULE DOC]
# Defines DTensorSpec, the internal description of how a DTensor is laid out
# (device mesh, per-mesh-dim placements, optional tensor metadata, and an
# explicit shard order for tensor dims sharded over several mesh dims), along
# with TensorMeta, ShardOrderEntry / ShardOrder, and the
# _StridedShardNotDecodableError exception. Includes conversion between
# _StridedShard placements and shard orders, and helpers for local shapes and
# string formatting. No import-time side effects.
import hashlib
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast, NamedTuple

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    _is_shard_like,
    _MaskPartial,
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.utils._debug_mode import _stringify_shape
from torch.utils._dtype_abbrs import dtype_abbrs


# Defined here (not in placement_types.py) because decoding split_factor into
# a shard order is a DTensorSpec concern — placement_types doesn't know about
# shard orders.
class _StridedShardNotDecodableError(ValueError):
    # [CLASS DOC] HELPER: ValueError subclass signalling that the split_factor
    # values of _StridedShard placements do not correspond to any valid shard
    # order. No extra state.
    """Raised when _StridedShard split_factor cannot be decoded into a shard order."""


class ShardOrderEntry(NamedTuple):
    # [CLASS DOC]
    # Named tuple for one sharded tensor dim: tensor_dim and the non-empty tuple
    # mesh_dims of mesh dims it is sharded over, in application order. A tuple
    # of these entries (ShardOrder) is sorted by tensor_dim.
    """
    Represents how a single tensor dimension is sharded across mesh dimensions.

    Attributes:
        tensor_dim: The tensor dimension being sharded (e.g., 0, 1, 2 for a 3D tensor).
        mesh_dims: Tuple of mesh dimensions across which this tensor dimension is sharded,
                   in execution order. The first mesh dim is applied first, second is applied
                   second, etc. This tuple is guaranteed to be non-empty.

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_DISTRIBUTED)
        >>> # Tensor dim 1 sharded across mesh dim 2, then mesh dim 0
        >>> ShardOrderEntry(tensor_dim=1, mesh_dims=(2, 0))

        >>> # Tensor dim 0 sharded only on mesh dim 1
        >>> ShardOrderEntry(tensor_dim=0, mesh_dims=(1,))
    """

    tensor_dim: int
    mesh_dims: tuple[int, ...]  # guaranteed to be non-empty


# Type alias for the complete shard order specification
# A tuple of ShardOrderEntry, one per sharded tensor dimension
#
# Example:
#   shard_order = (
#       ShardOrderEntry(tensor_dim=0, mesh_dims=(1,)),
#       ShardOrderEntry(tensor_dim=2, mesh_dims=(0, 3)),
#   )
#   This means:
#     - Tensor dimension 0 is sharded on mesh dimension 1
#     - Tensor dimension 2 is sharded on mesh dimension 0 first, then mesh dimension 3
ShardOrder = tuple[ShardOrderEntry, ...]


class TensorMeta(NamedTuple):
    # [CLASS DOC]
    # Minimal named tuple of tensor metadata (global shape, stride, dtype) used
    # only for sharding propagation.
    # simple named tuple to represent tensor metadata
    # intentionally to stay simple only for sharding
    # propagation purposes.
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype


# used internally to propagate the placements
@dataclass
class DTensorSpec:
    # [CLASS DOC]
    # Dataclass describing a DTensor's distribution: mesh, placements (one per
    # mesh dim, normalized to a tuple), tensor_meta (set only during sharding
    # propagation), shard_order (which mesh dims shard each tensor dim and in
    # what order) and use_strided_shard_as_shard_order. Invariant: when
    # _StridedShard placements encode the order, shard_order must be None and is
    # derived on demand; otherwise shard_order defaults to left-to-right order.
    # Caches its hash in _hash. Provides construction from a dim map, shard-order
    # conversion, local shape helpers and string formatting.
    mesh: DeviceMesh
    placements: tuple[Placement, ...]

    # tensor meta will only be set during sharding propagation
    tensor_meta: TensorMeta | None = None

    # When a tensor dimension is sharded across multiple mesh axes,
    # `shard_order` specifies the sequence in which these shardings are applied.
    # This order determines how tensor shards are mapped and distributed across
    # devices.
    #
    # Example:
    #   For a tensor of shape [8, 16] and a 3D device mesh, if dim 0 is sharded over
    #   mesh dim 1, and dim 1 is sharded over mesh dim 0 and then mesh dim 2,
    #   the shard_order would be:
    #     shard_order = (
    #         ShardOrderEntry(tensor_dim=0, mesh_dims=(1,)),
    #         ShardOrderEntry(tensor_dim=1, mesh_dims=(0, 2)),
    #     )
    shard_order: ShardOrder = None  # type: ignore[assignment]

    # When True, _StridedShard placements encode the shard order, and the
    # shard_order field must be left as None (it will be derived on demand).
    # Set explicitly to False to treat _StridedShard as a regular Shard,
    # e.g., in view propagation.
    use_strided_shard_as_shard_order: bool | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Dataclass hook: converts placements to a tuple; if
        # use_strided_shard_as_shard_order is unset, sets it True iff any
        # _StridedShard is present. When True and shard_order is also given,
        # raises ValueError (conflicting orders); when False and shard_order is
        # None, fills in the default shard order. Resets the cached hash.
        if not isinstance(self.placements, tuple):
            self.placements = tuple(self.placements)
        if self.use_strided_shard_as_shard_order is None:
            if any(isinstance(p, _StridedShard) for p in self.placements):
                self.use_strided_shard_as_shard_order = True
            else:
                self.use_strided_shard_as_shard_order = False
        if self.use_strided_shard_as_shard_order:
            if self.shard_order is not None:
                raise ValueError(
                    "DTensorSpec doesn't allow specify shard_order when "
                    "use_strided_shard_as_shard_order is True. This may result "
                    "in conflicting shard order."
                )
        else:
            if self.shard_order is None:
                self.shard_order = self.compute_default_shard_order(self.placements)

        self._hash: int | None = None

    @staticmethod
    def _normalize_placements_into_shard_order(
        placements: tuple[Placement, ...],
        mesh: DeviceMesh,
        use_strided_shard_as_shard_order: bool = True,
    ) -> tuple[tuple[Placement, ...], ShardOrder]:
        # [FUNCTION DOC] HELPER: Static method returning (placements, shard_order).
        # If use_strided_shard_as_shard_order, decodes _StridedShard placements
        # into a shard order (raising _StridedShardNotDecodableError if
        # impossible) and replaces each _StridedShard with a plain Shard of the
        # same dim. Otherwise returns placements unchanged with the default order.
        # If use_strided_shard_as_shard_order, it means the StridedShard/Shard
        # combinations should be interpreted as shard order.
        if use_strided_shard_as_shard_order:
            # _StridedShard in placements, try check if it can be decoded as shard order
            shard_order = DTensorSpec._maybe_convert_StridedShard_to_shard_order(
                placements, mesh
            )
            if shard_order is None:
                raise _StridedShardNotDecodableError(
                    f"_StridedShard placements {placements} cannot be decoded "
                    "into a corresponding shard_order"
                )
            normalized_placements = tuple(
                [
                    p if not isinstance(p, _StridedShard) else Shard(p.dim)
                    for p in placements
                ]
            )
            return normalized_placements, shard_order
        else:
            return placements, DTensorSpec.compute_default_shard_order(placements)

    @staticmethod
    def compute_default_shard_order(
        placements: tuple[Placement, ...],
    ) -> ShardOrder:
        # [FUNCTION DOC]
        # Static method: builds the default ShardOrder by scanning mesh dims left
        # to right and grouping shard-like placements by tensor dim; entries are
        # sorted by tensor dim. Raises AssertionError if a shard dim is negative
        # (placements must be normalized).
        """
        Compute the default shard order from placements.

        Returns a ShardOrder where each ShardOrderEntry maps a tensor dimension
        to the mesh dimensions it's sharded on, in left-to-right order.

        Args:
            placements: Tuple of Placement objects representing how a tensor is
                distributed across mesh dimensions.
        """
        # follow default left-to-right device order if shard_order is not specified
        tensor_dim_to_mesh_dims: defaultdict[int, list[int]] = defaultdict(list)
        mesh_ndim = len(placements)
        for mesh_dim in range(mesh_ndim):
            if _is_shard_like(placements[mesh_dim]):
                placement = placements[mesh_dim]
                shard_dim = placement.dim  # pyrefly: ignore [missing-attribute]
                if shard_dim < 0:
                    raise AssertionError(
                        f"Shard dim {shard_dim} in placements {placements} must be normalized"
                    )
                tensor_dim_to_mesh_dims[shard_dim].append(mesh_dim)

        # Convert dict into ShardOrderEntry tuples
        default_shard_order = tuple(
            ShardOrderEntry(tensor_dim=key, mesh_dims=tuple(value))
            for key, value in sorted(tensor_dim_to_mesh_dims.items())
            if value
        )
        return default_shard_order

    @staticmethod
    def _convert_shard_order_to_StridedShard(
        shard_order: ShardOrder, placements: tuple[Placement, ...], mesh: DeviceMesh
    ) -> tuple[Placement, ...]:
        # [FUNCTION DOC] HELPER: Static method rewriting the Shard placements named
        # by shard_order into Shard or _StridedShard. For each mesh dim in an
        # entry, split_factor is the product of sizes of mesh dims that come
        # earlier in the entry but have a larger mesh index; factor 1 gives plain
        # Shard, otherwise _StridedShard with that factor. Raises ValueError if a
        # referenced placement is not exactly a Shard. Returns a new tuple.
        """
        Convert ShardOrder to placements with _StridedShard.

        This function converts a ShardOrder specification into a tuple of Placement objects,
        using _StridedShard when a tensor dimension is sharded across multiple mesh dimensions
        in a non-default order. The split_factor of each _StridedShard is determined by the
        product of mesh dimension sizes that appear earlier in the shard order but later in
        the placement tuple.

        Args:
            shard_order: ShardOrder specification indicating which tensor dimensions are
                sharded on which mesh dimensions and in what execution order.
            placements: Tuple of Placement objects that does not contain _StridedShard.
            mesh: DeviceMesh containing the size information for each mesh dimension.

        Returns:
            Updated tuple of Placement objects with Shard or _StridedShard placements.

        Algorithm:
            For each ShardOrderEntry in shard_order:
              - For each mesh dimension in the entry's mesh_dims (in order):
                - Calculate split_factor as the product of mesh sizes for all mesh dimensions
                  that appear:
                  1. Earlier in the shard order (lower index in mesh_dims), and
                  2. Later in the placement tuple (higher mesh dimension index)
                - If split_factor == 1: use normal Shard
                - Otherwise: use _StridedShard with the calculated split_factor

        Example:
            >>> # xdoctest: +SKIP("Requires DeviceMesh")
            >>> # Tensor dimension 0 sharded on mesh dims [2, 0, 1] in that order
            >>> # mesh = DeviceMesh([4, 3, 2])  # sizes: mesh[0]=4, mesh[1]=3, mesh[2]=2
            >>> shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 0, 1)),)
            >>> placements = (Shard(0), Shard(0), Shard(0))
            >>> # For mesh_dim=2 (index 0 in mesh_dims): no earlier dims, split_factor=1
            >>> #   -> placements[2] = Shard(0)
            >>> # For mesh_dim=0 (index 1 in mesh_dims): mesh_dim=2 is earlier and has index 2>0
            >>> #   -> split_factor = mesh.size(2) = 2
            >>> #   -> placements[0] = _StridedShard(0, split_factor=2)
            >>> # For mesh_dim=1 (index 2 in mesh_dims): mesh_dim=2 is earlier and has index 2>1
            >>> #   -> split_factor = mesh.size(2) = 2
            >>> #   -> placements[1] = _StridedShard(0, split_factor=2)
            >>> # Result: (_StridedShard(0, sf=2), _StridedShard(0, sf=2), Shard(0))
        """
        placements_list = list(placements)
        for entry in shard_order:
            tensor_dim = entry.tensor_dim
            mesh_dims = entry.mesh_dims
            for idx in range(len(mesh_dims)):
                # TODO(zpcore): split_factor from `view` and `shard order`
                # should be able to be multiplied into one. Need to loosen the
                # condition here.
                mesh_dim = mesh_dims[idx]
                if type(placements[mesh_dim]) is not Shard:
                    raise ValueError(
                        f"Only Shard placement can be converted to _StridedShard, "
                        f"found {placements[mesh_dim]} in {placements=}."
                    )
                split_factor = math.prod(
                    mesh.size(i) for i in mesh_dims[:idx] if i > mesh_dim
                )
                if split_factor == 1:
                    # use normal Shard
                    placements_list[mesh_dim] = Shard(tensor_dim)
                else:
                    placements_list[mesh_dim] = _StridedShard(
                        tensor_dim, split_factor=split_factor
                    )
        return tuple(placements_list)

    @staticmethod
    def _maybe_convert_StridedShard_to_shard_order(
        placements: tuple[Placement, ...], mesh: DeviceMesh
    ) -> ShardOrder | None:
        # [FUNCTION DOC] HELPER: Static method inverting the strided-shard
        # conversion. Without _StridedShard it returns the default order.
        # Otherwise walks mesh dims right to left, and for each shard-like
        # placement inserts its mesh dim into that tensor dim's ordering at the
        # position where the product of earlier mesh sizes equals its split factor
        # (1 for plain Shard). Returns None if no position matches. Raises
        # ValueError for placements other than shard-like, Replicate, Partial or
        # _MaskPartial.
        """
        Try to convert _StridedShard placements to ShardOrder.

        This is the inverse of `_convert_shard_order_to_StridedShard`. It reconstructs the shard
        order by examining the split_factor of each _StridedShard and determining its position
        in the execution order. If the _StridedShard configuration cannot be represented as a
        valid ShardOrder (i.e., there's no shard order that produces the observed split_factors),
        this function returns None.

        Args:
            placements: Tuple of Placement objects that may contain _StridedShard.
            mesh: DeviceMesh containing the size information for each mesh dimension.

        Returns:
            ShardOrder if conversion is possible, None otherwise. For placements without
            _StridedShard, returns the default shard order.

          Algorithm:
              1. If no _StridedShard in placements, return default shard order
              2. Create an empty list for each tensor dimension to represent mesh dim ordering
              3. Iterate through placements in reverse order (right to left):
                 - For each Shard/_StridedShard on a tensor dimension:
                   - Extract its split_factor (1 for Shard, split_factor for _StridedShard)
                   - Find the position in mesh_dims_order where accumulated_sf equals split_factor
                   - accumulated_sf is the product of mesh sizes of mesh dimensions that appear
                     earlier in mesh_dims_order (lower indices)
                   - Insert mesh_dim at the found position
              4. If no valid position found for any split_factor, return None (unable to convert)
              5. Construct ShardOrderEntry for each tensor dimension from mesh_dims_order

        Example:
            >>> # xdoctest: +SKIP("Requires DeviceMesh")
            >>> # mesh = DeviceMesh([4, 3, 2])  # sizes: mesh[0]=4, mesh[1]=3, mesh[2]=2
            >>> # placements = (_StridedShard(0, sf=2), _StridedShard(0, sf=2), Shard(0))
            >>> # Process tensor_dim=0 from right to left:
            >>> #   - mesh_dim=2: Shard(0) with sf=1
            >>> #     Try position 0: accumulated_sf=1, matches! Insert at position 0
            >>> #     Current mesh_dims_order order: [2]
            >>> #   - mesh_dim=1: _StridedShard(0, sf=2) with sf=2
            >>> #     Try position 0: accumulated_sf=1, no match
            >>> #     Try position 1: accumulated_sf=1*mesh.size(2)=2, matches! Insert at position 1
            >>> #     Current mesh_dims_order order: [2, 1]
            >>> #   - mesh_dim=0: _StridedShard(0, sf=2) with sf=2
            >>> #     Try position 0: accumulated_sf=1, no match
            >>> #     Try position 1: accumulated_sf=1*mesh.size(2)=2, matches! Insert at position 1
            >>> #     Final mesh_dims_order order: [2, 0, 1]
            >>> # Result: ShardOrder((ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 0, 1)),))
            >>> # This means: first shard on mesh_dim=2, then mesh_dim=0, then mesh_dim=1

        Note:
            This function validates that _StridedShard can be represented as a ShardOrder.
            Not all _StridedShard configurations are valid - the split_factor must match
            the product of mesh sizes in some execution order.
        """
        if not any(isinstance(p, _StridedShard) for p in placements):
            return DTensorSpec.compute_default_shard_order(placements)
        max_tensor_dim = max([i.dim for i in placements if _is_shard_like(i)]) + 1
        shard_order = []

        tensor_dim_to_mesh_dims_order: list[list[int]] = [
            [] for i in range(max_tensor_dim)
        ]
        for mesh_dim in reversed(range(len(placements))):
            cur_placement = placements[mesh_dim]
            if _is_shard_like(cur_placement):
                tensor_dim = cur_placement.dim
                mesh_dims_order = tensor_dim_to_mesh_dims_order[tensor_dim]
                cur_sf = 1
                if isinstance(cur_placement, _StridedShard):
                    cur_sf = cur_placement.split_factor
                accumulated_sf = 1
                find_order = False
                for i in range(len(mesh_dims_order) + 1):
                    if accumulated_sf == cur_sf:
                        mesh_dims_order.insert(i, mesh_dim)
                        find_order = True
                        break
                    if i < len(mesh_dims_order):
                        accumulated_sf *= mesh.size(mesh_dims_order[i])
                if not find_order:
                    # _StridedShard is not convertible to ShardOrder
                    return None
            else:
                if not isinstance(cur_placement, Replicate | Partial | _MaskPartial):
                    raise ValueError(
                        f"Unsupported placement type {type(cur_placement)} encountered in "
                        f"{placements}; expected Replicate, Partial, or _MaskPartial."
                    )
        for tensor_dim in range(max_tensor_dim):
            if len(tensor_dim_to_mesh_dims_order[tensor_dim]) > 0:
                shard_order.append(
                    ShardOrderEntry(
                        tensor_dim=tensor_dim,
                        mesh_dims=tuple(tensor_dim_to_mesh_dims_order[tensor_dim]),
                    )
                )
        return tuple(shard_order)

    def _verify_shard_order(self, shard_order: ShardOrder) -> None:
        # [FUNCTION DOC] HELPER: Validates shard_order against the placements:
        # non-empty mesh dims, non-negative strictly increasing tensor dims,
        # in-range mesh dims whose placement is exactly Shard of that tensor dim,
        # and a total count equal to the number of Shard placements. Raises
        # AssertionError on violation. Skipped entirely if any _StridedShard exists.
        """Verify that the shard_order is valid and matches the placements."""
        total_shard = 0
        if any(isinstance(p, _StridedShard) for p in self.placements):
            # _StridedShard shard_order validation not yet supported;
            # the Shard-only checks below (line 390, 394) would fail.
            return
        prev_tensor_dim = -1
        for entry in shard_order:
            tensor_dim = entry.tensor_dim
            mesh_dims = entry.mesh_dims
            if len(mesh_dims) <= 0:
                raise AssertionError(f"shard_order {shard_order} has empty mesh dim")
            if tensor_dim < 0:
                raise AssertionError(
                    f"shard_order {shard_order} has invalid tensor dim {tensor_dim}"
                )
            if tensor_dim <= prev_tensor_dim:
                raise AssertionError("tensor dim should be sorted in shard_order")
            prev_tensor_dim = tensor_dim
            total_shard += len(mesh_dims)
            for mesh_dim in mesh_dims:
                if not (0 <= mesh_dim < len(self.placements)):
                    raise AssertionError(
                        f"shard_order {shard_order} has invalid mesh dim {mesh_dims}"
                    )
                if self.placements[mesh_dim] != Shard(tensor_dim):
                    raise AssertionError(
                        f"placement[{mesh_dim}] doesn't have a matching shard in shard_order"
                    )
        if total_shard != sum(1 for p in self.placements if isinstance(p, Shard)):
            raise AssertionError

    def __setattr__(self, attr: str, value: Any) -> None:
        # [FUNCTION DOC]
        # Intercepts attribute assignment: a non-None shard_order is validated
        # first (AssertionError if inconsistent with placements); the value is
        # stored; the cached hash is cleared when mesh, placements, tensor_meta
        # or shard_order change; and a non-None tensor_meta must be a TensorMeta
        # or fx TensorMetadata, else AssertionError.
        if attr == "shard_order" and value is not None:
            self._verify_shard_order(value)
        super().__setattr__(attr, value)
        # Make sure to recompute the hash in case any of the hashed attributes
        # change (though we do not expect `mesh`, `placements` or `shard_order`
        # to change)
        if hasattr(self, "_hash") and attr in (
            "mesh",
            "placements",
            "tensor_meta",
            "shard_order",
        ):
            self._hash = None
        # This assert was triggered by buggy handling for dict outputs in some
        # FX passes, where you accidentally iterate over a dict and try to put
        # keys into TensorMeta.  See https://github.com/pytorch/pytorch/issues/157919
        if attr == "tensor_meta" and value is not None:
            from torch.fx.passes.shape_prop import TensorMetadata

            # TODO: the TensorMetadata arises from
            # test/distributed/tensor/experimental/test_tp_transform.py::TensorParallelTest::test_tp_transform_e2e
            # but I actually can't reproduce it, maybe it is also a bug!
            if not isinstance(value, TensorMeta | TensorMetadata):
                raise AssertionError(repr(value))

    def _hash_key(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] HELPER: Returns the tuple of fields identifying this spec:
        # mesh, placements, shard_order and, when tensor_meta exists, its shape,
        # stride and dtype.
        """Return the tuple used for hashing. Used by both __hash__ and _stable_hash."""
        if self.tensor_meta is not None:
            return (
                self.mesh,
                self.placements,
                self.shard_order,
                self.tensor_meta.shape,
                self.tensor_meta.stride,
                self.tensor_meta.dtype,
            )
        return (self.mesh, self.placements, self.shard_order)

    def _hash_impl(self) -> int:
        # [FUNCTION DOC] HELPER: Computes the hash of the hash key. Must stay in
        # sync with equality if fields are added.
        # hashing and equality check for DTensorSpec are used to cache the sharding
        # propagation results. We only need to consider the mesh, placements, shape
        # dtype and stride.
        # Caveat: we need to keep this in mind and sync hash and eq if we add more
        # fields to them.
        return hash(self._hash_key())

    def __hash__(self) -> int:
        # [FUNCTION DOC] Returns the cached hash, computing and caching it on first
        # use (lazy so Dynamo does not hash symbolic strides eagerly). The cache is
        # reset by attribute assignment.
        # We lazily cache the spec to avoid recomputing the hash upon each
        # use, where we make sure to update the hash when the `tensor_meta`
        # changes by overriding `__setattr__`. This must be lazy so that Dynamo
        # does not try to hash non-singleton `SymInt`s for the stride.
        if self._hash is None:
            self._hash = self._hash_impl()
        return self._hash

    def _stable_hash(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a hex digest (16-byte blake2b of the text of
        # the hash key) that is stable across processes, using the mesh's own
        # stable hash (after unwrapping a fake script object) in place of the mesh.
        # Used for AOT autograd caching.
        """
        Return a stable hash for AOT autograd caching.
        [See note: Tensor subclass stable hashing for AOT autograd cache]
        """
        from torch._library.fake_class_registry import maybe_unwrap_fake_script_object

        # Get hash key, but replace mesh with its stable hash
        key = self._hash_key()
        # First element is mesh, replace with its stable hash
        mesh = maybe_unwrap_fake_script_object(self.mesh)
        stable_key = (mesh._stable_hash(),) + key[1:]
        return hashlib.blake2b(repr(stable_key).encode(), digest_size=16).hexdigest()

    def _check_equals(self, other: object, skip_shapes: bool = False) -> bool:
        # [FUNCTION DOC] HELPER: True if other is a DTensorSpec with equal mesh,
        # placements and shard_order and consistent tensor metadata: if either
        # tensor_meta is None both must be equal as is; otherwise dtype must match,
        # and shape and stride too unless skip_shapes is True.
        if not (
            isinstance(other, DTensorSpec)
            and self.mesh == other.mesh
            and self.placements == other.placements
            and self.shard_order == other.shard_order
        ):
            return False
        if self.tensor_meta is None or other.tensor_meta is None:
            return self.tensor_meta == other.tensor_meta

        if skip_shapes:
            return self.tensor_meta.dtype == other.tensor_meta.dtype
        return (
            self.tensor_meta.shape == other.tensor_meta.shape  # type: ignore[union-attr]
            and self.tensor_meta.stride == other.tensor_meta.stride  # type: ignore[union-attr]
            and self.tensor_meta.dtype == other.tensor_meta.dtype  # type: ignore[union-attr]
        )

    def __eq__(self, other: object, /) -> bool:
        # [FUNCTION DOC] Equality including shapes and strides, via _check_equals.
        return self._check_equals(other)

    def __str__(self) -> str:
        # [FUNCTION DOC] Text of the form Spec(dtype shape (placements)), where the
        # placement text comes from format_shard_order_str and dtype/shape fall
        # back to unknown placeholders when tensor_meta is None.
        """
        human readable representation of the DTensorSpec
        """
        placement_str = self.format_shard_order_str(self.placements, self.shard_order)
        if self.tensor_meta is not None:
            tensor_shape = _stringify_shape(self.tensor_meta.shape)
            tensor_dtype = dtype_abbrs[self.tensor_meta.dtype]
        else:
            tensor_shape = "unknown shape"
            tensor_dtype = "unknown dtype"

        return f"Spec({tensor_dtype}{tensor_shape}({placement_str}))"

    @staticmethod
    def is_default_device_order(shard_order: ShardOrder) -> bool:
        # [FUNCTION DOC] Static method: True if every entry lists its mesh dims in
        # strictly increasing order; False if shard_order is None (for example
        # when _StridedShard encodes the order) or any entry is not increasing.
        """
        Check if the device order is the default left-to-right order.
        """
        if shard_order is None:
            # Missing `shard_order` attribute, possibly due to _StridedShard in
            # placements.
            return False
        for entry in shard_order:
            mesh_dims = entry.mesh_dims
            is_increasing = all(
                prev < nxt for prev, nxt in itertools.pairwise(mesh_dims)
            )
            if not is_increasing:
                return False
        return True

    @staticmethod
    def format_shard_order_str(
        placements: tuple[Placement, ...],
        shard_order: ShardOrder | None = None,
    ) -> str:
        # [FUNCTION DOC] Static method building the mesh-ordered placement text by
        # concatenating each placement's text; for shard-like placements whose
        # tensor dim is sharded over several mesh dims (per shard_order), appends
        # the position of that mesh dim in the order in brackets. Raises
        # AssertionError if a shard-like placement's mesh dim is missing from its
        # shard_order entry.
        """
        Format DTensor sharding information as a human-readable string.

        This method formats the sharding pattern in mesh-centric order, showing the placement
        for each mesh dimension sequentially. When a tensor dimension is sharded across multiple
        mesh dimensions, the order index indicates the execution sequence of the sharding operations.

        Args:
            placements: Tuple of placement objects for each mesh dimension.
            shard_order: Optional ShardOrder specifying the sharding order.

        Returns:
            String representation of the sharding pattern in mesh-centric format.

        Example:
            For a 3D tensor on a 2x2x2x2 mesh (16 devices) with::

                placements = [Partial(), Shard(1), Shard(1), Replicate()]
                shard_order = (ShardOrderEntry(tensor_dim=1, mesh_dims=(2, 1)),)

            Mesh configuration:
                - mesh_dim_0: Partial reduction (sum)
                - mesh_dim_1: Shard tensor dimension 1 (executed second, order index 1)
                - mesh_dim_2: Shard tensor dimension 1 (executed first, order index 0)
                - mesh_dim_3: Replicate

            Output: ``"PS(1)[1]S(1)[0]R"``

            Explanation:
                - ``P``: mesh dimension 0 has partial reduction
                - ``S(1)[1]``: mesh dimension 1 shards tensor dimension 1 (order index 1 means second)
                - ``S(1)[0]``: mesh dimension 2 shards tensor dimension 1 (order index 0 means first)
                - ``R``: mesh dimension 3 replicates

            The format follows mesh dimension order (0, 1, 2, 3), and when a tensor dimension
            is sharded across multiple mesh dimensions, the bracketed index shows the execution
            order: ``[0]`` is executed first, ``[1]`` is executed second, etc.
        """
        out_str = ""
        # native dtensor-style sharding representation: map from mesh
        # dim to tensor dim
        for mesh_dim, placement in enumerate(placements):
            if _is_shard_like(placement):
                if shard_order is not None:
                    for entry in shard_order:
                        tensor_dim = entry.tensor_dim
                        mesh_dims = entry.mesh_dims

                        if placement.dim == tensor_dim:
                            if mesh_dim not in mesh_dims:
                                raise AssertionError
                            if len(mesh_dims) > 1:
                                out_str += f"{placement}[{mesh_dims.index(mesh_dim)}]"
                            else:
                                # no need to show device order if the tensor dim is
                                # only sharded in one mesh dim
                                out_str += str(placement)
                            break
                else:
                    out_str += str(placement)
            else:
                out_str += str(placement)
        return out_str

    @property
    def shape(self) -> torch.Size:
        # [FUNCTION DOC] Property: global tensor shape from tensor_meta; raises
        # ValueError if tensor_meta is not set.
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.shape

    @property
    def stride(self) -> tuple[int, ...]:
        # [FUNCTION DOC] Property: global tensor stride from tensor_meta; raises
        # ValueError if tensor_meta is not set.
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.stride

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC] Property: tensor rank from tensor_meta; raises ValueError
        # if tensor_meta is not set.
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return len(self.tensor_meta.shape)

    @property
    def num_shards(self) -> int:
        # [FUNCTION DOC] Property: product of the mesh sizes over all mesh dims
        # whose placement is shard-like (1 if nothing is sharded).
        num_shards = 1
        for i, placement in enumerate(self.placements):
            if _is_shard_like(placement):
                num_shards *= self.mesh.size(i)
        return num_shards

    @property
    def device_mesh(self) -> DeviceMesh:
        # [FUNCTION DOC] Property: alias for the mesh field, matching DTensor.
        # simple aliasing for the mesh field, make some
        # checks that mixes DTensor/DTensorSpec easier
        return self.mesh

    @property
    def dim_map(self) -> list[int]:
        # [FUNCTION DOC] Property: list with one entry per tensor dim giving the
        # mesh dim it is sharded on, or -1 if not sharded. Needs tensor_meta
        # (ValueError otherwise). Raises ValueError if one tensor dim is sharded
        # on more than one mesh dim.
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. It simply return a list of ints
        where dim_map[i] denotes the sharding mapping to the mesh
        dimension, and len(dim_map) == dist_tensor.ndim
        dim_map[i] = -1: means tensor dim i replicate on mesh
        dim_map[i] = j: means tensor dim i shard on mesh dim j

        For example, we have a dist tensor that have the shape of
        [18, 20, 30], and device_mesh([0, 1, 2, 3]), placements:
        [Shard(1)], the dim_map of this placement would be:
        [-1, 0, -1]. This representation is pretty helpful during
        sharding propagation where we could know exactly each
        tensor dimension is sharded or not.

        Note that if placements contains `_Partial`, we have to
        explicitly deal with it, so that when we create a DTensorSpec
        with dim_map, we could properly record the pending sums.
        """
        # dims mapping of dist tensor sharding
        # return size of tensor ndim, -1 represent replicate
        # and int >=0 represent shard on that device mesh dim
        r = [-1] * self.ndim
        for i, placement in enumerate(self.placements):
            if _is_shard_like(placement):
                shard_dim = placement.dim
                if r[shard_dim] > -1:
                    raise ValueError(
                        f"Tensor dim {shard_dim} is already sharded on mesh dim {r[shard_dim]},"
                        " DTensor operator implementation does not support things like hybrid"
                        " sharding strategies yet (i.e. [Shard(0), Shard(0)])"
                    )
                r[shard_dim] = i
        return r

    @property
    def num_shards_map(self) -> list[int]:
        # [FUNCTION DOC] Property: list with one entry per tensor dim giving the
        # total number of shards of that dim (product of mesh sizes of all mesh
        # dims sharding it; 1 if unsharded). Needs tensor_meta.
        # KNOWN ISSUE: the docstring's first sentence says dim_map; it describes
        # num_shards_map.
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. Unlike `dim_map`, `num_shards_map`
        denotes how many shards each tensor dim has. Like `dim_map`:
            len(num_shards_map) == dist_tensor.ndim
            num_shards_map[i] = 1: means tensor dim i is not sharded
            num_shards_map[i] = j: means tensor dim i has j shards in total

        For example, we have a dist tensor of shape [18, 20, 30],
        a device_mesh ([[0, 1, 2, 3], [4, 5, 6, 7]]), and placements
        ([Shard(1), Shard(0)]), the num_shards_map of this distributed tensor
        would be: [4, 2, 1].
        """
        r = [1] * self.ndim
        for i, placement in enumerate(self.placements):
            if _is_shard_like(placement):
                r[placement.dim] *= self.mesh.size(i)

        return r

    @property
    def sums(self) -> list[int]:
        # [FUNCTION DOC] Property: list of mesh dim indices whose placement is
        # partial (pending reduction).
        """
        sums is a property we derive from `placements` of the
        distributed tensor. It simply return a list of ints where
        sums[i] denotes the pending sum (partial) on mesh dim i
        """
        return [
            idx
            for idx, placement in enumerate(self.placements)
            if placement.is_partial()
        ]

    @classmethod
    def from_dim_map(
        cls,
        mesh: DeviceMesh,
        dim_map: list[int],
        sums: list[int],
        tensor_meta: TensorMeta | None = None,
    ) -> "DTensorSpec":
        # [FUNCTION DOC] Classmethod building a spec from a dim map (mesh dim per
        # tensor dim, -1 for none) and a list of partial mesh dims: starts all
        # Replicate, sets Partial for sums, then Shard(tensor dim) at each mapped
        # mesh dim. Raises RuntimeError if a mesh dim is already sharded by another
        # tensor dim, or is both partial and sharded. Uses default shard order.
        """
        Construct a DTensorSpec from dim_map list and pending sum.

        Args:
            mesh (class:`DeviceMesh`): device mesh to be used in the DTensorSpec
            dim_map (List[int]): a list of integer that represents sharding on each
                tensor dimension, see `dim_map` property doc for details
            sums (List[int]): a list of integer that represents the dist tensor have
                pending sum on which device mesh dimension.
            tensor meta (TensorMeta): DTensor metadata

        Return:
            a class:`DTensorSpec` object
        """
        # by default replicate on device mesh dims
        placements: list[Placement] = [Replicate() for _ in range(mesh.ndim)]

        # find all mesh dims that need pending reductions
        for s in sums:
            placements[s] = Partial()

        for i, m in enumerate(dim_map):
            if m >= 0:
                placement = placements[m]
                if placement.is_shard():  # dim_map only produces Shard placements
                    placement = cast(Shard, placement)
                    raise RuntimeError(
                        f"DeviceMesh dimension can't be mapped to two dimension of the same tensor: {i} and {placement.dim}"
                    )
                elif placement.is_partial():
                    raise RuntimeError(
                        f"DeviceMesh dimension {m} cannot be both shard and partial!"
                    )
                placements[m] = Shard(i)

        return cls(mesh, tuple(placements), tensor_meta=tensor_meta)

    def is_replicated(self) -> bool:
        # [FUNCTION DOC] True if every placement is Replicate.
        """
        return True if the current DTensorSpec replicates on all mesh dims (devices)
        """
        return all(placement.is_replicate() for placement in self.placements)

    def is_sharded(self) -> bool:
        # [FUNCTION DOC] True if any placement is shard-like (Shard or _StridedShard).
        """
        return True if the current DTensorSpec uses Shard() or _StridedShard() placement on any mesh dims (devices)
        """
        return any(_is_shard_like(placement) for placement in self.placements)

    def shallow_copy_with_tensor_meta(
        self, tensor_meta: TensorMeta | None
    ) -> "DTensorSpec":
        # [FUNCTION DOC] Returns a new DTensorSpec with the same mesh and placements
        # and the given tensor_meta, preserving use_strided_shard_as_shard_order.
        # Raises AssertionError if tensor_meta is None.
        # KNOWN ISSUE: shard_order is not copied; it is recomputed (default order)
        # unless strided shards encode it.
        """
        Shallow copy the DTensorSpec with a new tensor_meta.
        """
        if tensor_meta is None:
            raise AssertionError("shallow copy with no tensor_meta!")
        return DTensorSpec(
            self.mesh,
            self.placements,
            tensor_meta=tensor_meta,
            use_strided_shard_as_shard_order=self.use_strided_shard_as_shard_order,
        )
