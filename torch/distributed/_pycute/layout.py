#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

# [MODULE DOC] Layout algebra for the vendored pycute: Layout (shape/stride pair of IntTuples) and operations coalesce, filter, composition,
# complement, left/right inverse, logical divide/product, zipped and tiled variants, and slicing. Adapted to lexicographic
# (right-most is innermost) order rather than CuTe's co-lexicographic order. Many operations accept a tuple of layouts or None to work by mode.
# No import-time side effects.
"""
Definition of CuTe Layouts and functions to manipulate them which works with the order
of lexicographic instead of co-lexicographic as implemented in the original layout.py
"""

from itertools import chain
from typing import TypeAlias
from typing_extensions import Self, TypeIs

from .int_tuple import (
    crd2idx,
    flatten,
    has_none,
    IntTuple,
    is_int,
    is_tuple,
    product,
    slice_,
    suffix_product,
)


# Type aliases
CoordinateType: TypeAlias = (
    int | IntTuple | tuple[object, ...] | None
)  # Input for slice_ and crd2idx functions


class LayoutBase:
    # [CLASS DOC] Marker base class used only so is_layout can recognize layouts; has no state or methods.
    pass


def is_layout(x: object) -> TypeIs["Layout"]:
    # [FUNCTION DOC] Type guard: true if x is an instance of LayoutBase.
    return isinstance(x, LayoutBase)


class Layout(LayoutBase):
    # [CLASS DOC] A CuTe layout: public attributes shape and stride (IntTuples with the same structure) defining a function from logical coordinates to integer indices.
    # Supports equality, len (rank of the top-level shape; 1 if shape is an int), call (coordinate to index, or slicing when a coordinate has None),
    # indexing by mode, size (domain size), cosize (codomain extent) and string forms.
    def __init__(self, _shape: IntTuple, _stride: IntTuple | None = None) -> None:
        # [FUNCTION DOC] Store shape; stride defaults to the suffix product of shape (compact lexicographic strides).
        self.shape = _shape
        if _stride is None:
            self.stride = suffix_product(self.shape)
        else:
            self.stride = _stride

    # operator ==
    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal only to another Layout with identical shape and stride (no normalization); returns a bool.
        if not isinstance(other, Layout):
            return False
        return self.shape == other.shape and self.stride == other.stride

    # operator len(L)  (len [rank] like tuples)
    def __len__(self) -> int:
        # [FUNCTION DOC] Return the number of top-level modes (1 when shape is a plain int).
        if is_tuple(self.shape):
            return len(self.shape)
        else:
            return 1

    # operator ()    (map coord to idx)
    def __call__(self, *args: CoordinateType) -> Self | int:
        # [FUNCTION DOC] Evaluate the layout. With no None in the arguments, maps the coordinate (the single argument, or the arguments as a tuple) to a linear index via crd2idx.
        # With None placeholders, slices shape and stride and returns the sub-layout (mirrors cute C++ operator call).
        """
        Map a logical coordinate to a linear index (Coord has no Underscore slice operators)
        OR
        Slice the layout and return the sublayout (Coord has an Underscore slice op)

        Follow the same behavior of `Layout::operator(Coord const&)` in cute C++
        """
        if has_none(args):
            if len(args) == 1:
                return Layout(slice_(args[0], self.shape), slice_(args[0], self.stride))
            else:
                return Layout(slice_(args, self.shape), slice_(args, self.stride))
        else:
            if len(args) == 1:
                return crd2idx(args[0], self.shape, self.stride)  # type: ignore[arg-type]
            else:
                return crd2idx(args, self.shape, self.stride)  # type: ignore[arg-type]

    # operator []    (get-i like tuples)
    def __getitem__(self, i: int) -> Self:
        # [FUNCTION DOC] Return mode i as a Layout. For an int shape only i equal to 0 is valid (raises AssertionError otherwise).
        if is_tuple(self.shape):
            return Layout(self.shape[i], self.stride[i])  # type: ignore[index]
        else:
            if i != 0:
                raise AssertionError
            return Layout(self.shape, self.stride)

    # size(layout)   Size of the domain
    def size(self) -> int:
        # [FUNCTION DOC] Size of the domain: product of shape.
        return product(self.shape)

    # cosize(layout)   Size of the codomain
    def cosize(self) -> int:
        # [FUNCTION DOC] Size of the codomain: index of the last domain element plus one (assumes non-negative strides).
        return self(self.size() - 1) + 1  # type: ignore[operator]

    # print and str
    def __str__(self) -> str:
        # [FUNCTION DOC] Format as shape:stride.
        return f"{self.shape}:{self.stride}"

    # error msgs and representation
    def __repr__(self) -> str:
        # [FUNCTION DOC] Format as a constructor-like string with shape and stride.
        return f"Layout({self.shape},{self.stride})"


# Type aliases
LayoutOrIntTuple: TypeAlias = Layout | IntTuple
LayoutProfile: TypeAlias = tuple[object, ...] | Layout | None
LayoutInput: TypeAlias = Layout | IntTuple | tuple[object, ...] | None


# Make Layout from a list of layouts (each layout its own mode in the result)
def make_layout(*layouts: Layout | tuple[Layout, ...]) -> Layout:
    # [FUNCTION DOC] Build a layout whose modes are the given layouts (one mode per layout). Accepts varargs of layouts or a single iterable of layouts.
    # Returns Layout(tuple of shapes, tuple of strides).
    if len(layouts) == 1 and not is_layout(layouts[0]):
        layouts = layouts[0]

    shape, stride = zip(*((a.shape, a.stride) for a in layouts))  # type: ignore[union-attr]
    return Layout(shape, stride)


# Size of the domain
def size(layout: LayoutOrIntTuple) -> int:
    # [FUNCTION DOC] Domain size of a Layout, or the product of an IntTuple.
    if is_layout(layout):
        return layout.size()
    return product(layout)


# Size of the codomain
def cosize(layout: Layout) -> int:
    # [FUNCTION DOC] Codomain size of a layout (delegates to the method).
    return layout.cosize()


# Layout coalesce -- flatten and combine as many modes as possible while preserving the int-to-int function
def coalesce(layout: Layout, profile: LayoutProfile = None) -> Layout:
    # [FUNCTION DOC] Flatten and merge as many modes as possible without changing the int-to-int mapping, processing right to left (lexicographic).
    # Drops size-1 modes and merges adjacent modes whose stride matches the inner extent. With a tuple profile, coalesces by mode for the profiled
    # modes and keeps the rest (raises AssertionError if the layout has fewer modes than the profile). Returns a Layout (a size-1 stride-0 layout if empty).
    if is_tuple(profile):
        if len(layout) < len(profile):
            raise AssertionError
        return make_layout(
            # pyrefly: ignore [bad-argument-type]
            chain(
                (coalesce(layout[i], profile[i]) for i in range(len(profile))),  # type: ignore[arg-type]
                (layout[i] for i in range(len(profile), len(layout))),
            )
        )

    result_shape = [1]
    result_stride = [0]
    # Since we now follow lexicographic order, we need to process from right to left.
    # And to make implementation more efficient, we append to the end of list and reverse it in the end.
    for shape, stride in zip(
        reversed(flatten(layout.shape)), reversed(flatten(layout.stride))
    ):
        # skip their shape-1s
        if shape == 1:
            continue
        # replace our shape-1 with anything
        elif result_shape[-1] == 1:
            result_shape[-1] = shape
            result_stride[-1] = stride
        # merge modes if the shape*stride match
        elif result_shape[-1] * result_stride[-1] == stride:
            result_shape[-1] = result_shape[-1] * shape
        # append a new mode
        else:
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 1:
        return Layout(result_shape[0], result_stride[0])
    else:
        result_shape.reverse()
        result_stride.reverse()
        return Layout(tuple(result_shape), tuple(result_stride))


# Layout filter -- replace all stride-0 modes with size-1 and then coalesce to remove them
def filter(layout: Layout, profile: LayoutProfile = None) -> Layout:
    # [FUNCTION DOC] Remove all modes of size 1 or stride 0 from the flattened layout, then coalesce. Returns Layout(1, 0) if nothing remains.
    # With a tuple profile, applies by mode (AssertionError if the layout has fewer modes than the profile).
    if is_tuple(profile):
        if len(layout) < len(profile):
            raise AssertionError
        return make_layout(
            # pyrefly: ignore [bad-argument-type]
            chain(
                (filter(layout[i], profile[i]) for i in range(len(profile))),  # type: ignore[arg-type]
                (layout[i] for i in range(len(profile), len(layout))),
            )
        )

    result_shape = []
    result_stride = []
    for shape, stride in zip(flatten(layout.shape), flatten(layout.stride)):
        # skip their shape-1s and stride-0s
        if not (shape == 1 or stride == 0):
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 0:
        return Layout(1, 0)
    else:
        return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


# Layout composition
# Use tuples-of-layouts to perform this operation by-mode and None as no-op
def composition(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    # [FUNCTION DOC] Compose layoutA after layoutB (result maps a coordinate through B, then A). None for B is a no-op; an int B is treated as a layout of that size;
    # a tuple B applies by mode. For a plain layout B, walks the coalesced A right to left, dividing out B's stride and taking up to B's size from each mode.
    # Raises AssertionError if mode sizes are not divisible against the running stride, or if A has fewer modes than a tuple B.
    if layoutB is None:
        return layoutA
    elif is_int(layoutB):
        return composition(layoutA, Layout(layoutB))
    elif is_tuple(layoutB):
        if len(layoutA) < len(layoutB):
            raise AssertionError
        return make_layout(
            # pyrefly: ignore [bad-argument-type]
            chain(
                (composition(layoutA[i], layoutB[i]) for i in range(len(layoutB))),  # type: ignore[arg-type]
                (layoutA[i] for i in range(len(layoutB), len(layoutA))),
            )
        )
    elif is_tuple(layoutB.shape):
        return make_layout(composition(layoutA, layoutB_i) for layoutB_i in layoutB)  # type: ignore[arg-type, attr-defined]

    if layoutB.stride == 0:
        return Layout(layoutB.shape, 0)
    else:
        result_shape = []
        result_stride = []
        rest_shape = layoutB.shape
        rest_stride = layoutB.stride
        flat_A = coalesce(layoutA)
        # when left layout is multi-dimensional sublayout, aka, self = (a,b,...,c):(x,y,...,z), layout = s:d,
        # for integral s and d means that we want:
        # (1) “remove” the first d elements from left, starting from rightmost. (This will increase the stride.)
        # (2) “keep” the first s of those strided elements. (This does not affect the stride.)
        # For example, if self = (6,2):(2,1), layout = (3:2)
        # Step 1: remove the first 2 elements from self with stride increase, i.e., (6,2):(2,1) -> (6,1):(2,2)
        # Step 2: keep the first 3 of those strided elements, i.e., (6,1):(2,2) -> (3,1):(2,2)
        # Because we are going lexicographically, we go through left layout from right to left.
        for curr_shape, curr_stride in zip(
            reversed(flatten(flat_A.shape)[1:]), reversed(flatten(flat_A.stride)[1:])
        ):
            if not (curr_shape % rest_stride == 0 or rest_stride % curr_shape == 0):  # type: ignore[operator]
                raise AssertionError
            new_shape = min(max(1, curr_shape // rest_stride), rest_shape)  # type: ignore[operator]

            if new_shape != 1:
                result_shape.append(new_shape)  # Append to end, will reverse later
                result_stride.append(rest_stride * curr_stride)

            rest_shape = rest_shape // new_shape  # type: ignore[operator]
            rest_stride = -(
                -rest_stride // curr_shape  # type: ignore[operator]
            )  # Python exclusive impl: "//" is always floor div so == ceil_div(abs(rest_stride), curr_shape) * signum(rest_stride)

        # When left has single-size sublayout or reach the last sublayout, aka, left = a:b, layout = s:d,
        # the result is rather trivial: left o layout = a:b o s:d = s:(b*d).
        # For example, if self = (6:2), layout = (3:2), the result is (3:(2*2)) = (3:4).
        if rest_shape != 1 or len(result_shape) == 0:
            result_shape.append(rest_shape)  # Append to end, will reverse later
            result_stride.append(rest_stride * flatten(flat_A.stride)[0])

        # Reverse the lists because we build lists in reverse order (append to end), this way it is more efficient.
        result_shape.reverse()
        result_stride.reverse()

        if len(result_shape) == 1:
            return Layout(result_shape[0], result_stride[0])  # type: ignore[arg-type]
        else:
            return Layout(tuple(result_shape), tuple(result_stride))  # type: ignore[arg-type]


# Layout complement
def complement(layout: LayoutOrIntTuple, max_idx: int = 1) -> Layout:
    # [FUNCTION DOC] Compute the complement of a layout (or int size) w.r.t. max_idx: a layout filling the gaps in the layout's image so that the pair spans max_idx.
    # Sorts modes by stride, requires them non-overlapping (AssertionError otherwise, skipped for symbolic values), appends a final ceil-divided mode,
    # reverses to lexicographic order and coalesces.
    if is_int(layout):
        return complement(Layout(layout))

    result_shape = []
    result_stride = []
    current_idx = 1

    sorted_DS = sorted(zip(flatten(layout.stride), flatten(layout.shape)))  # type: ignore[union-attr]
    for stride, shape in sorted_DS:
        if stride == 0 or shape == 1:
            continue

        in_bound = current_idx <= shape * stride
        # To support symbolic value which can't be evaluated now
        if (type(in_bound) is bool) and not in_bound:
            raise AssertionError

        result_shape.append(stride // current_idx)
        result_stride.append(current_idx)
        current_idx = shape * stride

    result_shape.append((max_idx + current_idx - 1) // current_idx)  # ceil_div
    result_stride.append(current_idx)
    # This is different from original pycute implementation, because we want to follow the lexicographic order here
    # where the right-most dimension is the innermost dimension (smallest stride).
    result_shape.reverse()
    result_stride.reverse()

    return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


# Layout right inverse
def right_inverse(layout: LayoutOrIntTuple | None) -> Layout | None:
    # [FUNCTION DOC] Compute the right inverse (a layout f with layout(f(i)) == i over the contiguous prefix of the image). None gives None; an int becomes Layout(int).
    # Sorts modes by stride and stops at the first stride gap; returns a coalesced Layout.
    if layout is None:
        return None
    elif is_int(layout):
        return Layout(layout)

    result_shape = []
    result_stride = []
    current_idx = 1

    flat_shape = flatten(layout.shape)  # type: ignore[union-attr]
    flat_stride = flatten(layout.stride)  # type: ignore[union-attr]
    sorted_DSA = sorted(zip(flat_stride, flat_shape, suffix_product(flat_shape)))  # type: ignore[arg-type]
    for stride, shape, rstride in sorted_DSA:
        if shape == 1:
            continue
        if current_idx != stride:
            break

        result_shape.append(shape)
        result_stride.append(rstride)
        current_idx = shape * stride

    result_shape.reverse()
    result_stride.reverse()
    return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


# Layout left inverse
def left_inverse(layout: LayoutOrIntTuple | None) -> Layout | None:
    # [FUNCTION DOC] Compute the left inverse as the right inverse of the layout made from the complement followed by the layout. None gives None; an int becomes Layout(int).
    if layout is None:
        return None
    elif is_int(layout):
        return Layout(layout)
    return right_inverse(make_layout(complement(layout), layout))  # type: ignore[arg-type]


# Split a layout by the composition of B and the "rest"
# Use tuples-of-layouts to perform this operation by-mode and None as no-op
def logical_divide(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    # [FUNCTION DOC] Split layoutA by layoutB: compose A with a layout made of B and its complement w.r.t. the size of A. None is a no-op, an int B is converted to a layout,
    # a tuple B applies by mode (AssertionError if A has fewer modes).
    if layoutB is None:
        return layoutA
    elif is_int(layoutB):
        return logical_divide(layoutA, Layout(layoutB))
    elif is_tuple(layoutB):
        if len(layoutA) < len(layoutB):
            raise AssertionError
        return make_layout(
            # pyrefly: ignore [bad-argument-type]
            chain(
                (
                    logical_divide(layoutA[i], layoutB[i])  # type: ignore[arg-type]
                    for i in range(len(layoutB))
                ),
                (layoutA[i] for i in range(len(layoutB), len(layoutA))),
            )
        )

    return composition(
        layoutA,
        make_layout(layoutB, complement(layoutB, size(layoutA))),
    )


# Reproduce a layoutA over a layoutB
# Use tuples-of-layouts to perform this operation by-mode and None as no-op
def logical_product(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    # [FUNCTION DOC] Repeat layoutA over layoutB: a layout of A followed by the complement of A (sized to size of A times cosize of B) composed with B. None is a no-op; a tuple B applies by mode.
    # KNOWN ISSUE: the int case calls logical_divide instead of logical_product, which looks like a copy-paste bug.
    if layoutB is None:
        return layoutA
    elif is_int(layoutB):
        return logical_divide(layoutA, Layout(layoutB))
    elif is_tuple(layoutB):
        if len(layoutA) < len(layoutB):
            raise AssertionError
        return make_layout(
            # pyrefly: ignore [bad-argument-type]
            chain(
                (
                    logical_product(layoutA[i], layoutB[i])  # type: ignore[arg-type]
                    for i in range(len(layoutB))
                ),
                (layoutA[i] for i in range(len(layoutB), len(layoutA))),
            )
        )

    return make_layout(
        layoutA,
        composition(complement(layoutA, size(layoutA) * cosize(layoutB)), layoutB),
    )


# Gather the modes from a hierarchical logical_divide or logical_product
def hier_unzip(
    # [FUNCTION DOC] HELPER: Hierarchically apply a splitter (logical_divide or logical_product, which must return a rank-2 layout) and gather results into two modes:
    # all tile parts first, all rest parts second. None for B gives a layout of (trivial, A); a tuple B recurses per mode (AssertionError if A has fewer modes).
    splitter: object,
    layoutA: Layout,
    layoutB: LayoutInput,
) -> Layout:
    if layoutB is None:
        return make_layout(Layout(1, 0), layoutA)
    elif is_tuple(layoutB):
        if len(layoutA) < len(layoutB):
            raise AssertionError
        # A layout with shape ((A,a),(B,b),(C,c))
        split = make_layout(
            hier_unzip(splitter, layoutA[i], layoutB[i])  # type: ignore[arg-type]
            for i in range(len(layoutB))
        )
        # Gather to shape ((A,B,C,...),(a,b,c,...,y,z))
        return make_layout(
            make_layout(split[i][0] for i in range(len(layoutB))),  # type: ignore[arg-type]
            make_layout(
                chain(  # type: ignore[arg-type]
                    (split[i][1] for i in range(len(layoutB))),
                    (layoutA[i] for i in range(len(layoutB), len(layoutA))),
                )
            ),
        )

    # splitter must return a rank-2 layout
    return splitter(layoutA, layoutB)  # type: ignore[operator]


# Apply logical divide hierarchically and gather the split modes into two modes
def zipped_divide(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    # [FUNCTION DOC] Hierarchical logical divide with the split modes gathered into two modes.
    return hier_unzip(logical_divide, layoutA, layoutB)


# Perform logical divide hierarchically and gather tiles (B-layouts) into a new mode
def tiled_divide(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    # [FUNCTION DOC] Zipped divide, then the second mode is unpacked so each rest mode becomes its own top-level mode after the first.
    result = zipped_divide(layoutA, layoutB)
    return make_layout([result[0]] + [result[1][i] for i in range(len(result[1]))])  # type: ignore[arg-type]


# Apply logical product hierarchically and gather the split modes into two modes
def zipped_product(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    # [FUNCTION DOC] Hierarchical logical product with the split modes gathered into two modes.
    return hier_unzip(logical_product, layoutA, layoutB)


# Perform logical product hierarchically and gather tiles (B-layouts) into a new mode
def tiled_product(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    # [FUNCTION DOC] Zipped product, then the second mode is unpacked so each rest mode becomes its own top-level mode after the first.
    result = zipped_product(layoutA, layoutB)
    return make_layout([result[0]] + [result[1][i] for i in range(len(result[1]))])  # type: ignore[arg-type]


def slice_and_offset(crd: tuple[object, ...], layout: Layout) -> tuple[Layout, int]:
    # [FUNCTION DOC] Slice layout by a coordinate containing None placeholders. Returns a pair: the sliced sub-layout, and the integer offset of the coordinate.
    return (
        Layout(slice_(crd, layout.shape), slice_(crd, layout.stride)),
        crd2idx(crd, layout.shape, layout.stride),  # type: ignore[arg-type]
    )
