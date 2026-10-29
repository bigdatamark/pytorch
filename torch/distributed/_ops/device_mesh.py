# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC] Defines and registers custom ops in the device_mesh library namespace used to trace DeviceMesh behavior under PT2:
# a runtime coordinate-on-dim op (with fake and eager impls), and custom ops that return a flattened sub-mesh or a sliced sub-mesh.
# Import-time side effect: defines the library op schema and registers the fake/impl/custom-op implementations.

import torch
from torch import SymInt

from ..device_mesh import DeviceMesh


# Register custom operator
torch.library.define(
    "device_mesh::_runtime_compute_coordinate_on_dim",
    "(Tensor full_mesh, int index) -> SymInt",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.register_fake("device_mesh::_runtime_compute_coordinate_on_dim")
def _runtime_compute_coordinate_on_dim_fake(
    full_mesh: torch.Tensor, index: int
) -> SymInt:
    # [FUNCTION DOC] Fake (meta) implementation of the coordinate op: returns a new unbacked SymInt bounded to 0..mesh size minus 1
    # on the requested mesh dim (full_mesh has an extra leading batch dim, so the dim is looked up at index+1).
    # When tracing under dynamo, the fresh symbol is appended to the shape env's ignorable unbacked symbols list (mutation).
    from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

    ctx = torch._custom_op.impl.get_ctx()
    shape_env = ctx._shape_env

    # Bypass allow_dynamic_output_shape_ops check by directly creating the symint.
    # This is intentional - the coordinate is always valid and bounded.
    sz = shape_env.create_unbacked_symint()

    # Apply size constraints - coordinate is bounded by mesh size on the given dimension.
    # The full_mesh tensor has an extra batch dimension at the front, so the actual
    # mesh dimensions start at index 1. mesh.size(index) = full_mesh.size(index + 1)
    mesh_size = full_mesh.size(index + 1)
    _constrain_range_for_size(
        sz, min=0, max=mesh_size - 1 if isinstance(mesh_size, int) else None
    )

    try:
        # Check if we're currently tracing in dynamo (as opposed to AOT or export).
        in_dynamo = torch._dynamo.symbolic_convert.InstructionTranslator.current_tx()
    except AttributeError:
        in_dynamo = False

    if in_dynamo:
        # During dynamo tracing, distributed ops are treated as atomic - so the
        # rank SymInt may be computed but not traced into the graph (e.g., it
        # affects tensor values but not shapes). Mark it as ignorable here;
        # when we decompose these ops later (after dynamo), we'll create fresh
        # SymInts that do get traced.
        shape_env.ignorable_fresh_unbacked_symbols.append(sz.node._expr)

    return sz


@torch.library.impl(
    "device_mesh::_runtime_compute_coordinate_on_dim", "CompositeExplicitAutograd"
)
def _runtime_compute_coordinate_on_dim_impl(full_mesh: torch.Tensor, index: int) -> int:
    # [FUNCTION DOC] Eager implementation: looks up the current process rank, derives the mesh tensor from full_mesh, computes the rank's mesh
    # coordinates and returns the coordinate on dimension index. Raises AssertionError if the rank is not part of the mesh.
    rank = torch.distributed.get_rank()
    mesh = DeviceMesh._get_mesh_tensor_from_full_mesh(full_mesh)
    mesh_coords = DeviceMesh._compute_coordinates_from_mesh(mesh, rank)
    if mesh_coords is None:
        raise AssertionError
    return mesh_coords[index]


def _get_flattened_submesh_impl(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    # [FUNCTION DOC] HELPER: Return the flattened sub-mesh of mesh over the given mesh_dims (looked up through the redistribute module's layout based lookup).
    # Raises ValueError if no such flattened mesh exists.
    from torch.distributed.tensor._redistribute import (
        _get_flattened_mesh_by_layout_impl,
    )

    result = _get_flattened_mesh_by_layout_impl(mesh, tuple(mesh_dims))
    if result is None:
        raise ValueError(f"No flattened mesh found for mesh_dims={mesh_dims} on {mesh}")
    return result


@torch.library.custom_op("device_mesh::_get_flattened_submesh", mutates_args=())
def _get_flattened_submesh(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    # [FUNCTION DOC] Custom op (no mutated args) returning the flattened sub-mesh for mesh_dims; delegates to the shared impl.
    return _get_flattened_submesh_impl(mesh, mesh_dims)


@_get_flattened_submesh.register_fake
def _get_flattened_submesh_fake(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    # [FUNCTION DOC] Fake registration for the flattened sub-mesh op; runs the same real impl.
    return _get_flattened_submesh_impl(mesh, mesh_dims)


def _get_submesh_impl(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    # [FUNCTION DOC] HELPER: Slice mesh by the dim names at positions mesh_dims and return the sub-mesh (single name gives a one-dim mesh, several names a multi-dim mesh).
    # Raises ValueError if the mesh has no dim names.
    all_dim_names = mesh._mesh_dim_names
    if all_dim_names is None:
        raise ValueError(f"Cannot slice mesh without dim names: {mesh}")
    dim_names = tuple(all_dim_names[i] for i in mesh_dims)
    if len(dim_names) == 1:
        return mesh[dim_names[0]]
    return mesh[dim_names]


@torch.library.custom_op("device_mesh::_get_submesh", mutates_args=())
def _get_submesh(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    # [FUNCTION DOC] Custom op (no mutated args) returning the sub-mesh for mesh_dims; delegates to the shared impl.
    return _get_submesh_impl(mesh, mesh_dims)


@_get_submesh.register_fake
def _get_submesh_fake(mesh: DeviceMesh, mesh_dims: list[int]) -> DeviceMesh:
    # [FUNCTION DOC] Fake registration for the sub-mesh op; runs the same real impl.
    return _get_submesh_impl(mesh, mesh_dims)
