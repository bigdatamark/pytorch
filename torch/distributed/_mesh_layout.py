# [MODULE DOC] Internal bookkeeping for DeviceMesh using CuTe-style layouts (shape/stride pairs over integer ranks).
# Defines _FlatLayout (one normalized, coalesced, 1-D-like layout) and _MeshLayout (a tuple of _FlatLayout axes
# representing a whole mesh), built on the pycute helpers (composition, complement, coalesce, ...).
# Used to compute flatten/unflatten/slice results and the rank groups for communicators. No import-time side effects.
"""
Definition of CuTe inspired Layouts for DeviceMesh internal bookkeeping and functions to manipulate them
"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import NoReturn, overload

import torch
from torch.distributed._pycute import (
    as_tuple,
    coalesce,
    complement,
    composition,
    flatten,
    IntTuple,
    is_int,
    is_tuple,
    Layout,
    make_layout,
    match_structure,
    suffix_product,
)


@dataclass(frozen=True)
class _FlatLayout:
    # [CLASS DOC] Immutable (frozen dataclass) normalized layout of one mesh dimension: attributes shape and stride are flat tuples of ints.
    # The constructor coalesces and flattens the input so equivalent layouts compare equal; size-1 dims vanish and a
    # trivial layout becomes the empty (0-dimensional) one. Length and indexing are deliberately disabled.
    # Key methods: to_pycute, numel, composition, complement, all_ranks_from_zero, global_ranks, check_sorted, check_orthogonal.
    """
    A canonical CuTe layout for a single dimension of a DeviceMesh

    Utility class for representing an integer layout by borrowing ideas from CuTe Layout Algebra.
    See https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md for more details.

    Each layout is represented as a list of sizes and strides. We use it as a way for mechanical bookkeeping
    of the integers such as ranks in a SPMD mesh, and the transformation on top of it.

    Lots of methods of layout like coalesce, composition, complement, etc. are borrowed from pycute.
    https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/python/pycute/layout.py#L137,L257

    Note this is a CuTe-inspired layout, because CuTe uses co-lexicographic way in linearization while PyTorch
    is using lexicographic. So even though the CuTe documentation can still be referenced, the implementation will be
    different from that of PyCute's.

    This layout is _not_ itself subdivided into multiple dimensions. It might
    internally sometimes use multidimensional tuple to represent "irregular"
    layouts (e.g., flattening non-adjacent dims), but this should be considered
    an opaque implementation detail.

    This class guarantees that all equivalent layouts are encoded as the same
    normalized representation, and thus compare equal. This is achieved by
    flattening and coalescing compatible adjacent dimensions (which includes
    removing all dimensions of size 1).

    """

    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def __init__(self, shape: IntTuple, stride: IntTuple | None = None) -> None:
        # [FUNCTION DOC] Build a normalized layout from shape and optional stride (default: suffix product of shape, i.e. row-major).
        # Raises TypeError if shape or stride is neither int nor tuple, ValueError if their nesting structures do not match.
        # Coalesces and flattens the result; a single size-1/stride-0 dim becomes the empty layout. Sets attributes via the
        # object-level setter because the dataclass is frozen.
        if not is_tuple(shape) and not is_int(shape):
            raise TypeError(f"shape must be a tuple or int, got {type(shape)}")
        stride = stride if stride is not None else suffix_product(shape)
        if not is_tuple(stride) and not is_int(stride):
            raise TypeError(f"stride must be a tuple or int, got {type(stride)}")
        if not match_structure(shape, stride):
            raise ValueError(f"sizes {shape} and strides {stride} don't match")

        coalesced_layout = coalesce(Layout(shape, stride))
        flat_shape = flatten(coalesced_layout.shape)
        flat_stride = flatten(coalesced_layout.stride)

        # pycute will preserve a size=1 dim if it's the only remaining dim, but
        # we prefer to stick to the Tensor convention and make it 0-dimensional
        if flat_shape == (1,) and flat_stride == (0,):
            flat_shape = ()
            flat_stride = ()

        # Set attributes using object.__setattr__ since frozen=True
        object.__setattr__(self, "shape", flat_shape)
        object.__setattr__(self, "stride", flat_stride)

    def __len__(self) -> NoReturn:
        # [FUNCTION DOC] HELPER: Always raises RuntimeError; the length of the internal representation is intentionally not exposed.
        raise RuntimeError(
            "You should never need to know the length of the internal representation of a FlatLayout"
        )

    def __getitem__(self, i: int) -> NoReturn:
        # [FUNCTION DOC] HELPER: Always raises RuntimeError; indexing into the internal representation is intentionally not exposed.
        raise RuntimeError(
            "You should never need to index into the internal representation of a FlatLayout"
        )

    def to_pycute(self) -> Layout:
        # [FUNCTION DOC] Convert to a pycute Layout; the empty layout maps to the trivial layout of size 1 and stride 0.
        if not self.shape:
            return Layout(1, 0)
        return Layout(self.shape, self.stride)

    def numel(self) -> int:
        # [FUNCTION DOC] Return the total number of elements (product of shape; 1 for the empty layout).
        return math.prod(self.shape)

    def composition(self, layout: "_MeshLayout") -> "_MeshLayout":
        # [FUNCTION DOC] Compose this layout with another (self applied after layout) via pycute composition, per top-level axis of the argument.
        # Takes a _MeshLayout and returns a _MeshLayout whose axes are the normalized per-axis results.
        # Result picks indices from this layout's output space, selected by the right-hand layout.
        """
        By-dimension composition allows one layout to "select from" or "filter through" another layout.
        Think of it as function composition: (self ∘ layout)(input) = self(layout(input))
        between two layouts. This function is a wrapper of pycute's composition.

        Mental model about how to understand the composition logic:
        - The LEFT layout (self) defines the "output space" - what indices are possible
        - The RIGHT layout (layout parameter) acts as a "selector" - which specific indices to pick
        - The composition only generates indices that the left layout could originally produce,
          but the right layout determines which indices to be picked.
        - The stride of the composition layout will not be smaller than the stride of the right layout,
          because when picking the indices the composition will at least follow the right layout's stride
          to move forward.

        Example:
          self = (6,2):(2,1)      # sizes=(6,2), strides=(2,1)
          layout = (3:2)          # sizes=(3,), stride=(2,)
          self o layout = (3:2)

        Returns:
          Layout being composed.
        """
        result = composition(self.to_pycute(), layout.to_pycute())
        result_axes = [
            _FlatLayout(shape, stride)
            for shape, stride in zip(as_tuple(result.shape), as_tuple(result.stride))
        ]
        return _MeshLayout(result_axes)

    def complement(self, world_size: int) -> "_FlatLayout":
        # [FUNCTION DOC] Return the complement layout relative to world_size: the layout that, repeated with this one, covers world_size ranks.
        # Delegates to pycute complement and normalizes the result. Errors from pycute (incompatible layout/world_size) propagate.
        """
        Compute the "complement layout" relative to a given world_size.
        A complement layout fills in the "missing" factor so that: self repeat a layout of complement(self, world_size)
        will get a complete world_size. We use ⊗ to denote the repeat operation.

        Example:
          self = (4:1)   # size=4, stride=1
          world_size = 8
          Then:
            complete needed factor = 8 / 4 = 2
            complement(self, 8) = (2:1)

          Together they form:
            (4:1) ⊗ (2:1) = (4,2):(2,1)
          which has world_size = 4 * 2 = 8, as required.

        In distributed terms, complement() is often used to derive the "other"
        rank grouping when splitting processes into 2D meshes.

        For a visualized explanation, see https://x.com/ezyang/status/1962364978393981433/
        """
        result = complement(self.to_pycute(), world_size)
        return _FlatLayout(result.shape, result.stride)

    def all_ranks_from_zero(self) -> list[int]:
        # [FUNCTION DOC] Enumerate every rank offset the layout generates, iterating coordinates in row-major order of the shape and
        # summing coordinate times stride. Returns a list of ints (a single 0-length-product entry list for the empty layout).
        """
        This function computes the all ranks specified by the layout starting from zero.

        How it works:
        1. we enumerate every possible coordinate (like a nested for-loop).
        If sizes = (2, 3), we get the following coordinates:
            (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)

        2. For each coordinate, we compute a linear rank index as:
            all_ranks_from_zero = sum(coord[i] * strides[i] for i in range(ndim))

        Example A:
        sizes = (2, 3)        # 2 rows, 3 cols
        strides = (3, 1)        # row-major layout
        coords = (0,0) -> 0*3 + 0*1 = 0
                 (0,1) -> 0*3 + 1*1 = 1
                 (0,2) -> 0*3 + 2*1 = 2
                 (1,0) -> 1*3 + 0*1 = 3
                 (1,1) -> 1*3 + 1*1 = 4
                 (1,2) -> 1*3 + 2*1 = 5
        result = [0, 1, 2, 3, 4, 5]

        Example B:
        sizes = (2, 3)
        strides = (1, 2)        # non-standard / strided layout
        coords = (0,0) -> 0*1 + 0*2 = 0
                 (0,1) -> 0*1 + 1*2 = 2
                 (0,2) -> 0*1 + 2*2 = 4
                 (1,0) -> 1*1 + 0*2 = 1
                 (1,1) -> 1*1 + 1*2 = 3
                 (1,2) -> 1*1 + 2*2 = 5
        result = [0, 2, 4, 1, 3, 5]
        """
        return [
            sum(c * s for c, s in zip(coord, self.stride))
            for coord in product(*(range(s) for s in self.shape))
        ]

    def global_ranks(self, world_size: int) -> list[list[int]]:
        # [FUNCTION DOC] Compute global rank groups for a world of world_size ranks: one list per offset taken from the complement layout,
        # each containing that offset added to every rank of this layout. Used to build communicators. Returns a list of lists of ints.
        """
        Build global ranks specified by the layout via two-level ranks composition.

        The nested list forms the Cartesian product of all ranks for one layout and offset
        regarding filling up the world_size with the layout.
        The final global ranks are the addition of these two. The result is a
        list of lists: one sublist per layout. This rank list will be used to build
        the communicator underlying the layout and the given `world_size`.

        Example:
        world_size = 16
        self.size = 4
        self.stride = 1
        ranks = [0, 1, 2, 3]
        offsets = [0, 4, 8, 12]
        result = [
            [0+0, 0+1, 0+2, 0+3],  # → [0, 1, 2, 3]
            [4+0, 4+1, 4+2, 4+3],  # → [4, 5, 6, 7]
            [8+0, 8+1, 8+2, 8+3],  # → [8, 9, 10,11]
            [12+0, 12+1, 12+2, 12+3],  # → [12,13,14,15]
        ]
        """
        return [
            [offset + rank for rank in self.all_ranks_from_zero()]
            for offset in self.complement(world_size).all_ranks_from_zero()
        ]

    def check_sorted(self) -> bool:
        # [FUNCTION DOC] Return True if the strides are already in non-increasing (descending) order.
        return tuple(sorted(self.stride, reverse=True)) == self.stride

    def check_orthogonal(self) -> bool:
        # [FUNCTION DOC] Return True if no two coordinates map to the same rank, i.e. dimensions do not overlap.
        # Layouts with fewer than two dims are trivially fine. Otherwise sorts dims by stride descending and requires each stride
        # to be a multiple of the next stride times the next size. Returns a bool.
        """
        Check if the layout has any overlap between the ranks it generates. If there is overlap,
        we return False, otherwise True.

        The layout is supposed to be injective i.e, aside from indice 0, indices from each
        dim of the layout must be non-overlapping.

        Example 1 - Valid (no overlap):
        Layout: sizes=(2,3), strides=(6,1)
        - Dim 1: stride=1, span=3*1=3, covers indices [0,1,2]
        - Dim 0: stride=6, span=2*6=12, covers indices [0,6]
        → No overlap since 6 > 3

        Example 2 - Invalid (overlap):
        Layout: sizes=(2,3), strides=(2,1)
        - Dim 1: stride=1, span=3*1=3, covers indices [0,1,2]
        - Dim 0: stride=2, span=2*2=4, covers indices [0,2]
        → Overlap! stride=2 < span=3, so indices [0,2] are duplicated

        Example 3 - Invalid (overlap):
        Layout: sizes=(4,2), strides=(1,1)
        - Dim 1: stride=1, span=4, covers indices [0,1,2,3]
        - Dim 0: stride=1, span=2, covers indices [0,1]
        → Overlap! stride is same for two dims, so indices [0,2] are duplicated

        Returns:
            bool: True if no overlap, False if overlap detected
        """
        if len(self.shape) < 2:
            return True
        stride, shape = zip(*sorted(zip(self.stride, self.shape), reverse=True))
        return all(
            stride[i] % (stride[i + 1] * shape[i + 1]) == 0
            for i in range(len(stride) - 1)
        )

    @property
    def sizes_and_strides(self) -> Iterator[tuple[int, int]]:
        # [FUNCTION DOC] Property: iterator of (size, stride) pairs, one per dimension.
        """Iterate over (size, stride) pairs for each dimension."""
        return zip(self.shape, self.stride)


@dataclass(frozen=True)
class _MeshLayout(Sequence[_FlatLayout]):
    # [CLASS DOC] Immutable (frozen dataclass) Sequence of _FlatLayout axes describing a full DeviceMesh; attribute axes is a tuple.
    # Supports len, iteration, integer indexing (returns an axis) and slicing (returns a new _MeshLayout).
    # Holds methods needing the multi-dimensional structure: top_level_sizes, numel, cosize, collapse, splice, remap_to_tensor.
    """
    A multi-dimensional structure consisting of a series of dimension-less layouts

    This class represents the layout of a full DeviceMesh, where the overall
    top-level ndim and "logical" shape are well defined, but each individual
    mesh axis is squashed and normalized into a canonical _FlatLayout.

    It only contains methods that need to make use of this multi-dimensional
    structure (i.e., which access the ndim or the top-level sizes). Everything
    else should go on _FlatLayout and accessed by first calling .collapse().

    """

    axes: tuple[_FlatLayout, ...]

    def __init__(self, axes: Sequence[_FlatLayout]) -> None:
        # [FUNCTION DOC] Store the given axes sequence as a tuple (set via the object-level setter since frozen).
        object.__setattr__(self, "axes", tuple(axes))

    @classmethod
    def from_sizes_strides(
        cls, sizes: tuple[int, ...], strides: tuple[int, ...] | None = None
    ) -> "_MeshLayout":
        # [FUNCTION DOC] Classmethod: build a layout with one single-dim axis per (size, stride) pair; strides default to the flattened
        # suffix product of sizes. Raises ValueError if sizes and strides differ in length.
        if strides is None:
            strides = flatten(suffix_product(sizes))
        if len(sizes) != len(strides):
            raise ValueError(
                f"sizes and strides must have the same length, got {len(sizes)} and {len(strides)}"
            )
        axes = tuple(_FlatLayout((s,), (d,)) for s, d in zip(sizes, strides))
        return cls(axes)

    def __len__(self) -> int:
        # [FUNCTION DOC] Return the number of axes.
        return len(self.axes)

    @overload
    def __getitem__(self, i: int) -> _FlatLayout: ...

    @overload
    def __getitem__(self, i: slice) -> "_MeshLayout": ...

    def __getitem__(self, i: int | slice) -> "_FlatLayout | _MeshLayout":
        # [FUNCTION DOC] Index with an int to get the axis, or with a slice to get a new _MeshLayout of the selected axes.
        if isinstance(i, slice):
            return _MeshLayout(self.axes[i])
        return self.axes[i]

    def __iter__(self) -> Iterator[_FlatLayout]:
        # [FUNCTION DOC] Iterate over the axes.
        return iter(self.axes)

    def to_pycute(self) -> Layout:
        # [FUNCTION DOC] Convert to a pycute Layout by concatenating each axis's pycute layout; zero axes gives the trivial layout (size 1, stride 0).
        if len(self.axes) == 0:
            return Layout(1, 0)
        return make_layout(*(axis.to_pycute() for axis in self.axes))

    @property
    def top_level_sizes(self) -> tuple[int, ...]:
        # [FUNCTION DOC] Property: tuple of the number of elements of each axis.
        return tuple(axis.numel() for axis in self.axes)

    def numel(self) -> int:
        # [FUNCTION DOC] Return the total number of elements (product of the top-level sizes).
        return math.prod(self.top_level_sizes)

    def cosize(self) -> int:
        # [FUNCTION DOC] Return the cosize (extent of the generated index range) from the pycute layout.
        return self.to_pycute().cosize()

    def collapse(self) -> _FlatLayout:
        # [FUNCTION DOC] Merge all axes into one coalesced _FlatLayout, forgetting the multi-axis structure.
        """
        Merge all axes into a single _FlatLayout.

        This is used to "forget" the multi-dimensional structure of this object
        and recover a "flat" (and coalesced) representation.
        """
        shapes = tuple(axis.shape for axis in self.axes)
        strides = tuple(axis.stride for axis in self.axes)
        return _FlatLayout(shapes, strides)

    def splice(self, start: int, end: int, layout: "_MeshLayout") -> "_MeshLayout":
        # [FUNCTION DOC] Out-of-place replacement of axes start through end (slice semantics) with the axes of the given layout.
        # Returns a new _MeshLayout; self is unchanged.
        """
        Replace (out-of-place) the start:end slice with the given list of layouts

        Returns the concatenation of self[:start] + layout + self[end:].
        """
        new_axes = list(self.axes)
        new_axes[start:end] = list(layout.axes)
        return _MeshLayout(new_axes)

    def remap_to_tensor(self, rank_map: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Map this layout onto the concrete rank_map tensor (must be 1-D, contiguous, with at least cosize elements) and
        # return a tensor of actual ranks with shape (number of repeats, *top_level_sizes). Computes the collapsed layout's complement
        # over rank_map's size, takes a strided view with complement dims first then layout dims, and reshapes.
        # Raises AssertionError if rank_map is not 1-D, not contiguous, or smaller than the cosize.
        """
        Leverage layout as an index for mesh tensor that re-maps the indexes after layout
        transformation to actual device ranks.

        With this method, the cute layout serves as the backend of indices bookkeeping for the
        mesh tensor when it comes to flatten, unflatten and slicing operations. The actual mesh
        tensor still represents the actual device assignment and ranks. We need this function
        to specify device allocation and create backend for a mesh. Although any transform of mesh tensors
        can be treated as a view or subset of mesh tensor, we do need to use the actual view or
        sub-tensor for DeviceMesh and its backend creation.

        The shape of the `rank_map` must be 1D and contiguous.

        Examples:

        Case 1 - Consecutive ranks, full world:
            original_mesh_tensor = [[0,1],[2,3]]  # 2x2 mesh, ranks 0-3
            world_size = 4
            layout = Layout(2:2)
            Return: [[0,2],[1,3]]

        Case 2 - Non-consecutive ranks:
            original_mesh_tensor = [[10,20],[30,40]]  # custom rank assignment
            world_size = 4
            layout = Layout(2:2)
            Return: [[[10,30],[20,40]]]

        Args:
            rank_map: The concrete mesh tensor with actual device ranks

        Returns:
            torch.Tensor: A tensor representing the actual device allocation from rank_map
        """
        if rank_map.ndim != 1:
            raise AssertionError
        if not rank_map.is_contiguous():
            raise AssertionError
        if rank_map.numel() < self.cosize():
            raise AssertionError

        self_layout = self.collapse()
        complement_layout = self_layout.complement(rank_map.numel())

        return rank_map.as_strided(
            complement_layout.shape + self_layout.shape,
            complement_layout.stride + self_layout.stride,
        ).reshape(-1, *self.top_level_sizes)
