# mypy: allow-untyped-defs
# [MODULE DOC] Shared utilities for pipeline parallelism: tensor and DTensor metadata records
# used to allocate and validate recv buffers, a DeviceMesh cache, the static/dynamic metadata
# inference mode enum, argument flattening helpers, stage/rank mapping generators, and
# metadata validation functions. Import-time side effect: creates the module logger only.
# Copyright (c) Meta Platforms, Inc. and affiliates

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import cast, Literal, overload, Protocol, TYPE_CHECKING, TypeAlias

import torch
from torch import fx
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_flatten, tree_unflatten


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.placement_types import Placement


logger = logging.getLogger(__name__)


class GetMeshCallback(Protocol):
    # [CLASS DOC] Protocol for a user-supplied callable that returns a DeviceMesh given mesh
    # dimension names and an optional mesh layout. Used by _MeshCache to rebuild meshes whose
    # descriptions arrived over P2P. Its call method is a one-line stub.
    """Callback to create/retrieve a DeviceMesh from its cache key components."""

    def __call__(
        self,
        mesh_dim_names: tuple[str, ...],
        mesh_layout: _MeshLayout | None,
    ) -> DeviceMesh: ...


# Key for mesh cache: (mesh_dim_names, mesh_layout)
# mesh_layout is the _MeshLayout object containing shape and stride (not actual ranks).
# This uniquely identifies a mesh within the same "universe" where all stages share
# the same rank tensor.
MeshCacheKey: TypeAlias = tuple[tuple[str, ...], _MeshLayout | None]


class PipeliningMetadataError(RuntimeError):
    # [CLASS DOC] Runtime error raised when pipeline stage metadata (shape, dtype, mesh, etc.)
    # is missing, mismatched or cannot be resolved during pipeline communication.
    """Raised on metadata mismatches during pipeline communication."""


@dataclass(frozen=True, slots=True)
class _TensorMeta:
    # [CLASS DOC] Immutable (frozen, slotted) record of a tensor's shape, stride, dtype and
    # requires_grad, used to allocate recv buffers and validate received tensors. For DTensors
    # the fields describe the local shard. Key methods: from_tensor, to_tensor, get_diff.
    """Tensor metadata for recv buffer allocation and validation.

    For plain tensors, these are the tensor's actual attributes.
    For DTensors, these are LOCAL shard attributes; global attributes
    are stored in :class:`_DTensorMeta`.
    """

    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> _TensorMeta:
        # [FUNCTION DOC] Static method. Builds a metadata record from a plain tensor (shape, stride,
        # dtype, requires_grad). Raises PipeliningMetadataError if given a DTensor.
        # KNOWN ISSUE: the docstring says TypeError is raised, but the code raises PipeliningMetadataError.
        """Create metadata from a plain tensor.

        Args:
            tensor: A plain ``torch.Tensor`` (not DTensor).

        Returns:
            Metadata capturing shape, stride, dtype, and requires_grad.

        Raises:
            TypeError: If ``tensor`` is a DTensor.
        """
        if isinstance(tensor, DTensor):
            raise PipeliningMetadataError(
                "Expected plain tensor, got DTensor. Use _DTensorMeta.from_dtensor instead."
            )
        return _TensorMeta(
            shape=tensor.shape,
            stride=tensor.stride(),
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

    def to_tensor(self, device: torch.device | str) -> torch.Tensor:
        # [FUNCTION DOC] Allocates an uninitialized strided tensor on the given device matching this
        # metadata. For floating-point dtypes, sets requires_grad to the recorded value; other dtypes
        # are left without grad. Returns the tensor.
        """Reconstruct a tensor on ``device`` from this metadata.

        Args:
            device: Target device for the tensor.

        Returns:
            An empty strided tensor on ``device``.
        """
        t = _make_tensor_from_meta(self, device)
        if t.is_floating_point():
            t.requires_grad_(self.requires_grad)
        return t

    def get_diff(self, other: _TensorMeta) -> list[str]:
        # [FUNCTION DOC] Compares this metadata with another and returns a list of human-readable
        # difference strings for shape, stride and dtype (empty list when equal). requires_grad is
        # deliberately ignored because it is a runtime concern, not a metadata invariant.
        """Return field-by-field differences with ``other``.

        Args:
            other: Metadata to compare against.

        Returns:
            List of human-readable difference strings (empty if equal).
        """
        if self == other:
            return []

        diffs = []
        if self.shape != other.shape:
            diffs.append(f"shape mismatch: {self.shape} vs {other.shape}")
        if self.stride != other.stride:
            diffs.append(f"stride mismatch: {self.stride} vs {other.stride}")
        if self.dtype != other.dtype:
            diffs.append(f"dtype mismatch: {self.dtype} vs {other.dtype}")
        # requires_grad is intentionally excluded: it is a runtime concern
        # determined by has_backward and grad context, not a metadata invariant.
        return diffs


@dataclass(frozen=True, slots=True)
class _DTensorMeta(_TensorMeta):
    # [CLASS DOC] Immutable extension of _TensorMeta for DTensors. Inherited fields describe the
    # local shard; extra fields hold global shape/stride, placements, mesh dimension names and
    # mesh layout. The DeviceMesh itself is not stored; it is looked up in a _MeshCache through
    # mesh_cache_key. Key methods: from_dtensor, to_dtensor, get_diff.
    """DTensor metadata extending :class:`_TensorMeta` with distribution info.

    Inherited fields (shape, stride, etc.) are LOCAL shard attributes.
    Additional fields capture global shape and placement information
    needed to reconstruct a :class:`DTensor` via ``DTensor.from_local()``.

    The :class:`DeviceMesh` is **not** stored (not serializable for P2P);
    it is looked up from :class:`_MeshCache` using
    ``(mesh_dim_names, mesh_layout)`` as the key.
    """

    # Global DTensor properties (for reconstruction)
    global_shape: torch.Size = field(default_factory=lambda: torch.Size([]))
    global_stride: tuple[int, ...] = field(default=())

    # DTensor distribution properties
    placements: tuple[Placement, ...] = field(
        default=()
    )  # e.g., (Shard(0), Replicate())

    # Mesh identification - used to look up the correct DeviceMesh from cache
    mesh_dim_names: tuple[str, ...] = field(default=())  # e.g., ("tp",) or ("dp", "tp")
    mesh_layout: _MeshLayout | None = field(
        default=None
    )  # _MeshLayout with shape/stride - uniquely identifies mesh within the same universe

    @staticmethod
    def from_dtensor(dtensor: DTensor) -> _DTensorMeta:
        # [FUNCTION DOC] Static method. Builds a _DTensorMeta from a DTensor, recording local shard
        # shape/stride, dtype, requires_grad, global shape/stride, placements, mesh dim names (empty
        # tuple if unnamed) and the mesh layout.
        """Create metadata from a DTensor.

        Args:
            dtensor: The DTensor to extract metadata from.

        Returns:
            Metadata capturing both local and global attributes.
        """
        device_mesh = dtensor.device_mesh

        return _DTensorMeta(
            # Local tensor attributes (for recv buffer allocation)
            shape=dtensor._local_tensor.shape,
            stride=dtensor._local_tensor.stride(),
            dtype=dtensor.dtype,
            requires_grad=dtensor.requires_grad,
            # Global DTensor attributes (for reconstruction)
            global_shape=dtensor.shape,
            global_stride=dtensor.stride(),
            # Distribution info
            placements=dtensor._spec.placements,
            mesh_dim_names=(
                tuple(device_mesh.mesh_dim_names) if device_mesh.mesh_dim_names else ()
            ),
            mesh_layout=device_mesh._layout,
        )

    @property
    def mesh_cache_key(self) -> MeshCacheKey:
        # [FUNCTION DOC] Property returning the pair of mesh dimension names and mesh layout, used as
        # the lookup key into a _MeshCache.
        """Cache key ``(mesh_dim_names, mesh_layout)`` for mesh lookup."""
        return (self.mesh_dim_names, self.mesh_layout)

    def to_dtensor(self, device: torch.device | str, mesh: DeviceMesh) -> DTensor:
        # [FUNCTION DOC] Rebuilds a DTensor on the given device and mesh from this metadata: allocates
        # an empty local tensor, wraps it via from_local with the stored placements and global
        # shape/stride without running consistency checks, then enables requires_grad if recorded and
        # the tensor is floating point. Returns the DTensor.
        """Reconstruct a DTensor on ``device`` with placements.

        Args:
            device: Target device for the local tensor.
            mesh: The ``DeviceMesh`` to attach.

        Returns:
            A DTensor on ``device``.
        """
        local_tensor = _make_tensor_from_meta(self, device)
        # Set requires_grad after from_local() so that the from_local
        # operation itself is not recorded in the autograd graph.
        dt = DTensor.from_local(
            local_tensor,
            device_mesh=mesh,
            placements=self.placements,
            shape=self.global_shape,
            stride=self.global_stride,
            run_check=False,
        )
        if self.requires_grad and dt.is_floating_point():
            dt = dt.requires_grad_(True)
        return cast(DTensor, dt)

    def get_diff(self, other: _TensorMeta) -> list[str]:
        # [FUNCTION DOC] Returns difference strings versus another metadata: base local-field
        # differences plus, when other is also a _DTensorMeta, global shape/stride, placements, mesh dim
        # names and mesh layout differences; if other is a plain _TensorMeta, adds a type-mismatch
        # entry. Returns an empty list when equal. Calls the base class method explicitly because
        # super() is unreliable with slotted dataclasses.
        """Return field-by-field differences, including DTensor-specific fields.

        Args:
            other: Metadata to compare against.

        Returns:
            List of human-readable difference strings (empty if equal).
        """
        if self == other:
            return []

        # Get base class differences (compares local shape/stride/dtype/requires_grad)
        # NOTE: Use explicit class call instead of super() because
        # @dataclass(slots=True) on both parent and child can break super().
        diffs = _TensorMeta.get_diff(self, other)

        # Add DTensor-specific comparisons if other is also _DTensorMeta
        if isinstance(other, _DTensorMeta):
            if self.global_shape != other.global_shape:
                diffs.append(
                    f"global_shape mismatch: {self.global_shape} vs {other.global_shape}"
                )
            if self.global_stride != other.global_stride:
                diffs.append(
                    f"global_stride mismatch: {self.global_stride} vs {other.global_stride}"
                )
            if self.placements != other.placements:
                diffs.append(
                    f"placements mismatch: {self.placements} vs {other.placements}"
                )
            if self.mesh_dim_names != other.mesh_dim_names:
                diffs.append(
                    f"mesh_dim_names mismatch: {self.mesh_dim_names} vs {other.mesh_dim_names}"
                )
            if self.mesh_layout != other.mesh_layout:
                diffs.append(
                    f"mesh_layout mismatch: {self.mesh_layout} vs {other.mesh_layout}"
                )
        else:
            diffs.append("type: _DTensorMeta vs _TensorMeta")

        return diffs


# Type alias for union of tensor metadata types
TensorMeta: TypeAlias = _TensorMeta | _DTensorMeta


# Not frozen: fields are populated incrementally during forward and
# backward metadata inference or from user provided static metadata
@dataclass(slots=True)
class _StageMeta:
    # [CLASS DOC] Mutable, slotted record of a stage's tensor metadata: inputs, outputs,
    # input_grads and output_grads (each optional). Fields are filled incrementally during
    # inference or from user-provided static metadata. Methods: has_any, has_dtensors,
    # is_complete_for_forward.
    """Consolidated tensor metadata for a pipeline stage's forward and backward passes."""

    inputs: tuple[TensorMeta, ...] | None = None
    outputs: tuple[TensorMeta, ...] | None = None
    input_grads: tuple[TensorMeta | None, ...] | None = None
    output_grads: tuple[TensorMeta | None, ...] | None = None

    def has_any(self) -> bool:
        # [FUNCTION DOC] Returns True if any of the four metadata fields is not None.
        """Check if any metadata field is populated."""
        return any(
            v is not None
            for v in [self.inputs, self.outputs, self.input_grads, self.output_grads]
        )

    def has_dtensors(self) -> bool:
        # [FUNCTION DOC] Returns True if any entry in the inputs or outputs metadata is a _DTensorMeta.
        """Check if any input/output metadata is DTensor type."""
        for metas in [self.inputs, self.outputs]:
            if metas and any(isinstance(m, _DTensorMeta) for m in metas if m):
                return True
        return False

    def is_complete_for_forward(self) -> bool:
        # [FUNCTION DOC] Returns True when both inputs and outputs metadata are populated.
        """Check if forward metadata is fully populated."""
        return self.inputs is not None and self.outputs is not None


@dataclass(frozen=True, slots=True)
class _StageForwardMeta:
    # [CLASS DOC] Immutable message payload carrying the forward output metadata of stage i, which
    # becomes the input metadata of stage i+1 during inference.
    """Forward metadata transmitted from stage *i* to stage *i+1* during inference."""

    forward_metas: tuple[TensorMeta, ...]  # Stage i's outputs → Stage i+1's inputs


@dataclass(frozen=True, slots=True)
class _StageBackwardMeta:
    # [CLASS DOC] Immutable message payload carrying stage i's input-gradient metadata, which
    # becomes stage i-1's output-gradient metadata during inference. Entries may be None.
    """Backward metadata transmitted from stage *i* to stage *i-1* during inference.

    Gradient placements may differ from forward activations
    (e.g., ``Replicate`` → ``Partial``).
    """

    backward_metas: tuple[
        TensorMeta | None, ...
    ]  # Stage i's input_grads → Stage i-1's output_grads


def _make_tensor_from_meta(
    meta: _TensorMeta,
    device: torch.device | str,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Returns an uninitialized tensor on the given device with exactly the
    # shape, stride and dtype from the metadata.
    """Create a tensor from metadata.

    Args:
        meta: Metadata with shape, stride, and dtype.
        device: Target device for the tensor.

    Returns:
        Empty tensor preserving the exact memory layout.
    """
    return torch.empty_strided(
        size=meta.shape,
        stride=meta.stride,
        dtype=meta.dtype,
        device=device,
    )


def _derive_grad_metas(
    tensor_metas: tuple[TensorMeta, ...],
) -> tuple[TensorMeta | None, ...]:
    # [FUNCTION DOC] HELPER: Maps a tuple of tensor metadata to gradient metadata: same shape,
    # stride and dtype with requires_grad False. Entries that do not require grad become None.
    # KNOWN ISSUE: DTensor metadata entries also become None (not mentioned in the docstring).
    """Derive gradient metadata from tensor metadata.

    Returns metadata with the same shape/stride/dtype but ``requires_grad=False``.
    Entries where the source has ``requires_grad=False`` become ``None``.
    """

    def derive_one(m: TensorMeta) -> TensorMeta | None:
        # [FUNCTION DOC] HELPER: Derives gradient metadata for one entry; None if it does not require
        # grad or is a DTensor meta, else a plain _TensorMeta without requires_grad.
        if not m.requires_grad:
            return None
        if isinstance(m, _DTensorMeta):
            return None
        return _TensorMeta(
            shape=m.shape,
            stride=m.stride,
            dtype=m.dtype,
            requires_grad=False,
        )

    return tuple(derive_one(m) for m in tensor_metas)


class _MeshCache:
    # [CLASS DOC] HELPER: Dict-backed cache of DeviceMesh objects keyed by mesh dim names and mesh
    # layout. Misses are resolved through an optional user callback. Assumes all stages share the
    # same rank tensor.
    """Cache for :class:`DeviceMesh` objects keyed by ``(mesh_dim_names, mesh_layout)``.

    Assumes all pipeline stages share the same rank tensor (true for
    TorchTitan-style frameworks where meshes derive from a common world).
    """

    def __init__(self, get_mesh_cb: GetMeshCallback | None = None) -> None:
        # [FUNCTION DOC] HELPER: Initializes an empty cache and stores the optional mesh-creation callback.
        self._cache: dict[MeshCacheKey, DeviceMesh] = {}
        self._get_mesh_cb = get_mesh_cb

    def get_mesh(self, key: MeshCacheKey) -> DeviceMesh:
        # [FUNCTION DOC] HELPER: Returns the cached mesh for the key; otherwise calls the callback to
        # create it and caches the result. Raises PipeliningMetadataError if there is no callback or
        # the callback returns None.
        """Return a cached mesh, or create one via the callback.

        Args:
            key: Cache key ``(mesh_dim_names, mesh_layout)``.

        Returns:
            The ``DeviceMesh``.

        Raises:
            PipeliningMetadataError: If not cached and no callback provided.
        """
        if key in self._cache:
            return self._cache[key]

        mesh_dim_names, mesh_layout = key

        if self._get_mesh_cb is None:
            raise PipeliningMetadataError(
                f"Mesh not found in cache for mesh_dim_names={mesh_dim_names}, "
                f"mesh_layout={mesh_layout}, and no get_mesh callback provided. "
                f"Provide a get_mesh callback or use DTensors in static mode."
            )

        mesh = self._get_mesh_cb(mesh_dim_names, mesh_layout)
        if mesh is None:
            raise PipeliningMetadataError(
                f"Mesh lookup failed: callback returned None for "
                f"mesh_dim_names={mesh_dim_names}, mesh_layout={mesh_layout}. "
                f"Ensure all stages use meshes from the same universe."
            )
        self._cache[key] = mesh
        return mesh

    def put(self, key: MeshCacheKey, mesh: DeviceMesh) -> None:
        # [FUNCTION DOC] HELPER: Stores a mesh under the given key, overwriting any existing entry.
        """Add a mesh to the cache."""
        self._cache[key] = mesh

    def update_from_tensors(self, tensors: tuple[torch.Tensor | None, ...]) -> None:
        # [FUNCTION DOC] HELPER: For each DTensor in the tuple (non-DTensors and None are skipped),
        # caches its device mesh under its dim names and layout key unless already present.
        """Extract and cache meshes from any :class:`DTensor` instances in *tensors*."""
        for tensor in tensors:
            if isinstance(tensor, DTensor):
                mesh = tensor.device_mesh
                dim_names = tuple(mesh.mesh_dim_names) if mesh.mesh_dim_names else ()
                mesh_layout = mesh._layout
                key = (dim_names, mesh_layout)
                if key not in self._cache:
                    self._cache[key] = mesh

    def __contains__(self, key: MeshCacheKey) -> bool:
        # [FUNCTION DOC] HELPER: Membership test on the cache keys.
        return key in self._cache

    def __len__(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the number of cached meshes.
        return len(self._cache)


# ============================================================================
# Inference mode enum
# ============================================================================


class InferenceMode(Enum):
    # [CLASS DOC] Enum of pipeline-wide metadata inference modes: STATIC (all metadata known, no
    # runtime inference) and DYNAMIC (at least one stage needs runtime inference). Chosen
    # collectively by the schedule so all ranks agree and avoid P2P hangs.
    """Pipeline-level metadata inference mode, determined collectively across all PP ranks.

    The mode is set by the schedule (not individual stages) because
    ``has_backward`` is only known at schedule creation time and all
    stages must agree to avoid P2P hangs.

    .. attribute:: STATIC

        All stages have sufficient metadata; runtime inference is skipped.

    .. attribute:: DYNAMIC

        At least one stage requires runtime metadata inference.
    """

    STATIC = "static"
    DYNAMIC = "dynamic"

    @classmethod
    def needs_dynamic(cls, meta: _StageMeta, stage_has_backward: bool) -> bool:
        # [FUNCTION DOC] Class method. Returns True if a stage needs dynamic inference: when forward
        # metadata is incomplete, or when it involves DTensors with a backward pass and either grad
        # metadata (input_grads or output_grads) is missing. Otherwise returns False.
        """Determine whether dynamic metadata inference is needed for a stage.

        Args:
            meta: Stage metadata from user-provided args.
            stage_has_backward: Whether a backward pass will be performed.

        Returns:
            ``True`` if dynamic inference is needed.
        """
        # Case 1: Forward metadata incomplete → needs DYNAMIC
        if not meta.is_complete_for_forward():
            return True

        # Case 2: No DTensors → STATIC is fine (bwd metadata derivable from fwd metadata)
        if not meta.has_dtensors():
            return False

        # Case 3: No backward needed → STATIC is fine (don't need grad metadata)
        if not stage_has_backward:
            return False

        # Case 4: DTensors with backward but missing ANY grad metadata → needs DYNAMIC
        # Both input_grads AND output_grads are required for static mode with DTensors
        if meta.input_grads is None or meta.output_grads is None:
            return True

        # Case 5: DTensors with complete grads → STATIC is fine
        return False


# ============================================================================
# Utility functions
# ============================================================================


def flatten_args(args, *, detach: bool = False):
    # [FUNCTION DOC] Flattens nested arguments into a flat list of leaves. With detach False
    # returns just the flat list. With detach True, tensors are detached but keep their
    # requires_grad flag, and returns a pair: the original nested structure rebuilt from the
    # detached leaves, and the flat detached list.
    """Flatten ``args`` into a list, optionally detaching tensors.

    Args:
        args: Nested arguments to flatten.
        detach: If ``True``, detach tensors while preserving ``requires_grad``.

    Returns:
        ``(new_args, flat_detached_args)`` when ``detach=True``;
        ``flat_args`` list otherwise.
    """
    flat_args, treespec = tree_flatten(args)

    if detach:
        flat_detached = [
            a.detach().requires_grad_(a.requires_grad)
            if isinstance(a, torch.Tensor)
            else a
            for a in flat_args
        ]
        new_args = tree_unflatten(flat_detached, treespec)
        return new_args, flat_detached

    return flat_args


# Backward compatibility alias
def flatten_args_detach(args):
    # [FUNCTION DOC] Deprecated backward-compatible alias; calls flatten_args with detach True.
    """Flatten and detach. Deprecated: use ``flatten_args(args, detach=True)``."""
    return flatten_args(args, detach=True)


def generate_stage_to_rank_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, int]:
    # [FUNCTION DOC] Builds a mapping from stage index to pipeline rank. Style "loop" assigns
    # stage modulo pp_size; style "v" zigzags ranks up then down in a V shape, keeping the rank
    # unchanged at each border. Raises ValueError if the V style num_stages is not divisible by
    # pp_size, or if the style is unknown.
    """
    Compute the stage id to rank mapping for either a looped or V-style schedule.

    Most commonly num_stages == pp_size * 2, but this function can be used to
    compute the mapping for any number of stages per rank.
    """
    mapping = {}
    if style == "loop":
        for stage_index in range(num_stages):
            mapping[stage_index] = stage_index % pp_size
    elif style == "v":
        if num_stages % pp_size != 0:
            raise ValueError(
                f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size} for V schedules"
            )

        rank_index = 0
        for stage_index in range(num_stages):
            mapping[stage_index] = rank_index
            # don't change rank if we are on the border (to keep v shape)
            if (stage_index + 1) % pp_size == 0:
                continue
            if (stage_index // pp_size) % 2 == 0:
                rank_index += 1
            else:
                rank_index -= 1
    else:
        raise ValueError(f"Style {style} is not supported.")
    return mapping


def generate_rank_to_stage_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, list[int]]:
    # [FUNCTION DOC] Inverts the stage-to-rank mapping into a mapping from rank to a sorted list
    # of stage indices. Propagates ValueError from the stage-to-rank generator.
    """
    Compute the rank to stage id mapping for either a looped or V-style schedule.

    This function inverts the stage_to_rank_mapping to get which stages are assigned to each rank.

    Returns a dictionary mapping rank -> list of stage indices assigned to that rank.
    """
    stage_to_rank = generate_stage_to_rank_mapping(pp_size, num_stages, style)

    # Invert the mapping: rank -> list of stages
    rank_to_stages: dict[int, list[int]] = {}
    for stage_id, rank in stage_to_rank.items():
        if rank not in rank_to_stages:
            rank_to_stages[rank] = []
        rank_to_stages[rank].append(stage_id)

    # Sort the stage lists for each rank to ensure consistent ordering
    for stages in rank_to_stages.values():
        stages.sort()

    return rank_to_stages


@dataclass(slots=True)
class PipeInfo:
    # [CLASS DOC] Slotted dataclass describing a split pipeline: the FX graph, the number of
    # stages, and whether the graph includes loss and backward.
    """
    Captures information for a pipeline (`Pipe` object).
    """

    graph: fx.Graph
    num_stages: int
    has_loss_and_backward: bool


# ============================================================================
# Metadata extraction helpers
# ============================================================================


def extract_tensor_meta(tensor: torch.Tensor) -> TensorMeta:
    # [FUNCTION DOC] Returns metadata for one tensor: a _DTensorMeta for a DTensor, otherwise a
    # _TensorMeta.
    """Extract metadata from a tensor.

    Handles both plain Tensor and DTensor correctly: DTensors are
    dispatched to ``_DTensorMeta.from_dtensor`` which captures local
    shard attributes plus global shape/placement info, while plain
    tensors use ``_TensorMeta.from_tensor``.

    Args:
        tensor: A plain tensor or DTensor.

    Returns:
        ``_TensorMeta`` for plain tensors, ``_DTensorMeta`` for DTensors.
    """
    if isinstance(tensor, DTensor):
        return _DTensorMeta.from_dtensor(tensor)
    else:
        return _TensorMeta.from_tensor(tensor)


@overload
def extract_tensor_metas(
    tensors: tuple[torch.Tensor, ...] | None,
    *,
    allow_none: Literal[False] = ...,
) -> tuple[TensorMeta, ...] | None: ...


@overload
def extract_tensor_metas(
    tensors: tuple[torch.Tensor | None, ...] | None,
    *,
    allow_none: Literal[True],
) -> tuple[TensorMeta | None, ...] | None: ...


def extract_tensor_metas(
    tensors: tuple[torch.Tensor | None, ...] | tuple[torch.Tensor, ...] | None,
    *,
    allow_none: bool = False,
) -> tuple[TensorMeta | None, ...] | None:
    # [FUNCTION DOC] Implementation of the overloaded extractor. Returns None if tensors is None;
    # otherwise a tuple of metadata for each tensor, with None entries for non-tensors. Raises
    # PipeliningMetadataError if a non-tensor entry exists while allow_none is False. The two
    # preceding overloads are type-only stubs.
    """Extract metadata from a tuple of tensors.

    Args:
        tensors: Tuple of tensors (may include ``None`` when ``allow_none=True``).
        allow_none: If ``True``, preserve ``None`` elements (for gradients).

    Returns:
        Tuple of ``TensorMeta``, or ``None`` if ``tensors`` is ``None``.

    Raises:
        PipeliningMetadataError: If ``None`` found and ``allow_none=False``.
    """
    if tensors is None:
        return None

    metas_with_none: list[TensorMeta | None] = []
    has_none = False
    for t in tensors:
        if isinstance(t, torch.Tensor):
            metas_with_none.append(extract_tensor_meta(t))
        else:
            has_none = True
            metas_with_none.append(None)
    if not allow_none and has_none:
        raise PipeliningMetadataError(
            "None values are not allowed in tensor metadata tuples. "
            "Use allow_none=True for optional values."
        )
    return tuple(metas_with_none)


def to_local_if_dtensor(tensor: torch.Tensor, detach: bool = False) -> torch.Tensor:
    # [FUNCTION DOC] Optionally detaches the tensor first (for plain tensors too), then returns the
    # local shard if it is a DTensor, or the tensor itself otherwise.
    """Convert a DTensor to its local shard, or return a plain tensor as-is.

    When ``detach=True``, the tensor is detached before conversion —
    this applies to both DTensors and plain tensors.

    Args:
        tensor: A tensor that may be a DTensor.
        detach: If ``True``, detach before ``to_local()`` to avoid
            redistribution during backward.

    Returns:
        The local tensor component.
    """
    maybe_detached_tensor = tensor.detach() if detach else tensor
    if isinstance(maybe_detached_tensor, DTensor):
        return maybe_detached_tensor.to_local()
    return maybe_detached_tensor


@overload
def validate_and_normalize_to_tuple(
    args: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | None,
    allow_none: Literal[False] = ...,
) -> tuple[torch.Tensor, ...] | None: ...


@overload
def validate_and_normalize_to_tuple(
    args: torch.Tensor
    | tuple[torch.Tensor | None, ...]
    | list[torch.Tensor | None]
    | None,
    allow_none: Literal[True] = ...,
) -> tuple[torch.Tensor | None, ...] | None: ...


def validate_and_normalize_to_tuple(
    args: torch.Tensor
    | tuple[torch.Tensor, ...]
    | tuple[torch.Tensor | None, ...]
    | list[torch.Tensor]
    | list[torch.Tensor | None]
    | None,
    allow_none: bool = False,
) -> tuple[torch.Tensor | None, ...] | tuple[torch.Tensor, ...] | None:
    # [FUNCTION DOC] Implementation of the overloaded normalizer. Returns None for None, wraps a
    # single tensor into a one-tuple, converts a list to a tuple. Raises PipeliningMetadataError for
    # None elements when allow_none is False, non-tensor elements, or an unsupported container type.
    """Normalize ``args`` to a tuple and validate that all elements are tensors.

    Args:
        args: A single tensor, tuple/list of tensors, or ``None``.
        allow_none: If ``True``, permit ``None`` elements (for gradients).

    Returns:
        Tuple of tensors, or ``None`` if ``args`` is ``None``.

    Raises:
        PipeliningMetadataError: On non-tensor values
            (or ``None`` when ``allow_none=False``).
    """
    if args is None:
        return None
    elif isinstance(args, torch.Tensor):
        return (args,)
    elif isinstance(args, (tuple, list)):
        for i, arg in enumerate(args):
            if arg is None:
                if not allow_none:
                    raise PipeliningMetadataError(
                        f"Stage arg[{i}] is None. "
                        f"Stage args must be tensors. Use kwargs for optional values."
                    )
                continue
            if not isinstance(arg, torch.Tensor):
                raise PipeliningMetadataError(
                    f"Stage arg[{i}] has type {type(arg).__name__}. "
                    f"All stage args must be tensors. Use kwargs for non-tensor inputs."
                )
        # Normalize list to tuple
        return tuple(args) if isinstance(args, list) else args
    else:
        raise PipeliningMetadataError(
            f"Stage args must be a tensor, tuple, or list of tensors, got {type(args).__name__}."
        )


# ============================================================================
# Validation functions
# ============================================================================


def validate_metadata(
    desc: str,
    expected: TensorMeta,
    actual: torch.Tensor | TensorMeta,
    *,
    raise_on_mismatch: bool = False,
    warn_on_mismatch: bool = False,
) -> list[str]:
    # [FUNCTION DOC] Compares expected metadata against a tensor (its metadata is extracted) or
    # metadata. If the two metadata types differ, returns a single type-mismatch message. Otherwise
    # returns the expected metadata's list of differences (empty if matching). On mismatch, raises
    # PipeliningMetadataError if raise_on_mismatch, and/or emits a UserWarning if warn_on_mismatch.
    """
    Compare expected metadata against actual tensor or metadata.

    This is the unified validation/comparison function that uses get_diff() from
    metadata classes. Works with both plain tensors and DTensors.

    For plain tensors: compares shape/stride/dtype/requires_grad.
    For DTensors: compares all properties including global shape and placements.

    Args:
        desc: Description for error/warning messages.
        expected: Expected tensor metadata (_TensorMeta or _DTensorMeta).
        actual: Actual tensor or metadata to compare against.
        raise_on_mismatch: If True, raise PipeliningMetadataError on mismatch.
        warn_on_mismatch: If True, issue a warning on mismatch.

    Returns:
        List of differences (empty if metadata matches).

    Raises:
        PipeliningMetadataError: If raise_on_mismatch=True and differences exist.
    """
    # Extract metadata if actual is a tensor
    if isinstance(actual, torch.Tensor):
        actual_meta = extract_tensor_meta(actual)
    else:
        actual_meta = actual

    # Type check: ensure both are same type for meaningful comparison
    if type(expected) is not type(actual_meta):
        type_diff = [
            f"type: expected {type(expected).__name__}, got {type(actual_meta).__name__}"
        ]
        if raise_on_mismatch:
            raise PipeliningMetadataError(f"{desc}: {type_diff[0]}")
        if warn_on_mismatch:
            warnings.warn(
                f"{desc}: Metadata type mismatch. {type_diff[0]}. "
                f"Using dynamically inferred metadata instead.",
                UserWarning,
                stacklevel=2,
            )
        return type_diff

    # Use get_diff() from the metadata class
    diffs = expected.get_diff(actual_meta)

    if diffs:
        if raise_on_mismatch:
            raise PipeliningMetadataError(f"{desc}: {'; '.join(diffs)}")
        if warn_on_mismatch:
            warnings.warn(
                f"{desc}: Metadata mismatch. {'; '.join(diffs)}. "
                f"Using dynamically inferred metadata instead.",
                UserWarning,
                stacklevel=2,
            )

    return diffs


def validate_tensors_metadata(
    desc: str,
    expected: tuple[TensorMeta | None, ...],
    actual: tuple[torch.Tensor | TensorMeta | None, ...],
    *,
    raise_on_mismatch: bool = True,
    warn_on_mismatch: bool = False,
) -> list[str]:
    # [FUNCTION DOC] Element-wise validation of a tuple of expected metadata against actual tensors
    # or metadata. A length mismatch or a None-versus-metadata mismatch produces a message that is
    # raised (PipeliningMetadataError) if raise_on_mismatch, warned if warn_on_mismatch, and
    # collected. Other entries are checked through validate_metadata. Returns the aggregated
    # difference list. Note the default of raise_on_mismatch is True here.
    """Validate metadata for a tuple of tensors element-wise.

    Args:
        desc: Description prefix for error/warning messages.
        expected: Tuple of expected metadata (may include ``None`` for grads).
        actual: Tuple of actual tensors or metadata to compare against.
        raise_on_mismatch: If ``True``, raise on the first mismatch.
        warn_on_mismatch: If ``True``, issue warnings for mismatches.

    Returns:
        Aggregated list of difference strings.

    Raises:
        PipeliningMetadataError: If lengths differ or on mismatch.
    """
    if len(expected) != len(actual):
        msg = f"{desc}: expected {len(expected)} tensors, got {len(actual)}"
        if raise_on_mismatch:
            raise PipeliningMetadataError(msg)
        if warn_on_mismatch:
            warnings.warn(msg, UserWarning, stacklevel=2)
        return [msg]

    all_diffs: list[str] = []
    for i, (exp, act) in enumerate(zip(expected, actual, strict=True)):
        if exp is None and act is None:
            continue
        if exp is None or act is None:
            msg = (
                f"{desc}[{i}]: expected {'None' if exp is None else 'metadata'}, "
                f"got {'None' if act is None else 'metadata'}"
            )
            if raise_on_mismatch:
                raise PipeliningMetadataError(msg)
            if warn_on_mismatch:
                warnings.warn(msg, UserWarning, stacklevel=2)
            all_diffs.append(msg)
            continue
        diffs = validate_metadata(
            f"{desc}[{i}]",
            exp,
            act,
            raise_on_mismatch=raise_on_mismatch,
            warn_on_mismatch=warn_on_mismatch,
        )
        all_diffs.extend(diffs)
    return all_diffs


def validate_static_arg_grad_correspondence(
    stage_index: int,
    args: tuple[torch.Tensor, ...],
    grads: tuple[torch.Tensor | None, ...],
    is_input: bool,
) -> None:
    # [FUNCTION DOC] Checks that forward tensors and their gradients correspond for static mode:
    # lengths must match, a tensor not requiring grad must have a None grad, a requiring-grad tensor
    # with a None grad only triggers a UserWarning, and a DTensor requiring grad needs a DTensor
    # or None grad. Violations of the hard rules raise PipeliningMetadataError.
    """
    Validate the args↔grads contract for static mode.

    Enforces four rules for each (arg, grad) pair:
      1. len(args) must equal len(grads).
      2. If arg.requires_grad is False, grad must be None.
      3. If arg.requires_grad is True and grad is None, emit a warning
         (this is legal at pipeline boundaries but may indicate a bug).
      4. If arg is a DTensor with requires_grad=True and grad is not None,
         grad must also be a DTensor.

    Args:
        stage_index: The stage index for error messages.
        args: Tuple of forward tensors.
        grads: Tuple of gradient tensors (can include None).
        is_input: True for input_args/input_grads, False for output_args/output_grads.

    Raises:
        PipeliningMetadataError: If any hard rule (1, 2, or 4) is violated.
    """
    kind = "input" if is_input else "output"
    args_name = f"{kind}_args"
    grads_name = f"{kind}_grads"

    # Rule 1: lengths must match
    if len(args) != len(grads):
        raise PipeliningMetadataError(
            f"Stage {stage_index}: {grads_name} length ({len(grads)}) does not match "
            f"{args_name} length ({len(args)}). Each forward tensor must have a "
            f"corresponding gradient entry (use None for tensors that don't require grad)."
        )

    for i, (arg, grad) in enumerate(zip(args, grads, strict=True)):
        # Rule 2: no grad for a non-differentiable arg
        if not arg.requires_grad and grad is not None:
            raise PipeliningMetadataError(
                f"Stage {stage_index}: {args_name}[{i}] has requires_grad=False, "
                f"but {grads_name}[{i}] is not None ({type(grad).__name__}). "
                f"Non-differentiable tensors must have None as their gradient entry."
            )

        # Rule 3: missing grad for a differentiable arg (warn, don't raise)
        if arg.requires_grad and grad is None:
            warnings.warn(
                f"Stage {stage_index}: {args_name}[{i}] has requires_grad=True, "
                f"but {grads_name}[{i}] is None. This is legal at pipeline boundaries "
                f"but may indicate a missing gradient.",
                UserWarning,
                stacklevel=2,
            )

        # Rule 4: DTensor arg must have DTensor grad
        if (
            isinstance(arg, DTensor)
            and arg.requires_grad
            and grad is not None
            and not isinstance(grad, DTensor)
        ):
            raise PipeliningMetadataError(
                f"Stage {stage_index}: {args_name}[{i}] is a DTensor with requires_grad=True, "
                f"but {grads_name}[{i}] is {type(grad).__name__}, expected DTensor or None. "
                f"DTensor gradients may have different placements than forward tensors."
            )
