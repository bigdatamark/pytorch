# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC]
# Defines the DTensor placement types: Shard, Replicate, Partial (the public
# ones, exported via __all__ together with the C++ base Placement) and the
# internal _StridedShard and _MaskPartial. These are Python subclasses of C++
# placement classes and implement the tensor-level mechanics: splitting and
# padding for uneven sharding, scatter/gather/reduce-scatter/all-to-all through
# functional collectives, and symbolic-shape-aware size computation.
# Import-time side effects: renames Placement's module, replaces the base
# Placement eq/hash/fx-repr hooks with error-raising partials, and registers all
# placement classes as opaque constant objects via _register_placements_as_opaque.
# KNOWN ISSUE: the base-class hooks are built with a helper that takes the method
# name as its first positional parameter and is also given it by keyword, so
# invoking them would fail with a TypeError instead of the intended RuntimeError.

import functools
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, cast, TypeGuard, TypeVar

import torch
import torch._C
import torch.distributed._functional_collectives as funcol
from torch._C._distributed import Placement
from torch.distributed import RankType
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._collective_utils import (
    fill_empty_tensor_to_shards,
    mesh_broadcast,
    mesh_scatter,
    pad_tensor,
    shard_dim_alltoall,
    unpad_tensor,
)
from torch.distributed.tensor._ops._mask_buffer import MaskBuffer
from torch.types import IntLikeType


__all__ = ["Placement", "Shard", "Replicate", "Partial"]

_RankTypeT = TypeVar("_RankTypeT", bound=RankType)
_STRIDED_SHARD_BASE_SPLIT_FACTOR: Any = torch._C._distributed.StridedShard.split_factor


# Appease TestPublicBindings.test_correct_module_names
Placement.__module__ = "torch.distributed.tensor.placement_types"


def _raise_error(method_name, _):
    # [FUNCTION DOC] HELPER: always raises RuntimeError stating that the named
    # placement method must be overridden by a subclass. Intended to be bound
    # (via partial) to the base Placement eq, hash and fx-repr hooks.
    raise RuntimeError(
        f"Placement method '{method_name}' should not be called as "
        "it should be overridden by the subclass"
    )


Placement.__eq__ = functools.partial(_raise_error, method_name="__eq__")
Placement.__hash__ = functools.partial(_raise_error, method_name="__hash__")
Placement.__fx_repr__ = functools.partial(_raise_error, method_name="__fx_repr__")


class _StridedShardOffsetMode(IntEnum):
    # [CLASS DOC] HELPER: integer enum selecting how _StridedShard materializes
    # shard offsets: FIRST (only the first offset as an int, or minus one when the
    # shard is empty), ALL (the full list of offsets), NONE (skip offsets).
    FIRST = 0
    ALL = 1
    NONE = 2


def _explicit_or_backed_hint(value: IntLikeType) -> int | None:
    # [FUNCTION DOC] HELPER: returns a concrete integer for value when one is
    # available: plain ints and int-convertible values directly; symbolic ints via
    # the shape environment's optimization hint, but only if all of their unbacked
    # symbols have hint overrides. Returns None when no hint can be determined
    # (no shape environment, unbacked symbols without overrides, or non-integer).
    from torch.fx.experimental.symbolic_shapes import (
        free_unbacked_symbols,
        optimization_hint,
    )

    if isinstance(value, int):
        return int(value)
    if not isinstance(value, torch.SymInt):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    shape_env = getattr(value.node, "shape_env", None)
    if shape_env is None:
        return None
    unbacked_symbols = free_unbacked_symbols(value.node.expr)
    if unbacked_symbols and not unbacked_symbols.issubset(
        shape_env.var_to_hint_override.keys()
    ):
        return None
    return int(optimization_hint(value, fallback=None))


def _guarded_hint_int(value: IntLikeType, *, reason: str) -> int:
    # [FUNCTION DOC] HELPER: converts value to a concrete int. For symbolic ints it
    # obtains a hint, raises RuntimeError (mentioning reason) if none exists, and
    # registers a deferred runtime assertion that the symbol equals the hint so the
    # specialization stays sound. Returns the integer.
    if isinstance(value, int):
        return int(value)
    if not isinstance(value, torch.SymInt):
        return int(value)
    hint = _explicit_or_backed_hint(value)
    if hint is None:
        raise RuntimeError(
            f"Cannot specialize symbolic {reason} without a hint: {value}"
        )
    _defer_symbolic_equality_assert(value, hint, reason=reason)
    return hint


def _defer_symbolic_equality_assert(
    value: IntLikeType, hint: int, *, reason: str
) -> None:
    # [FUNCTION DOC] HELPER: for a symbolic int with a shape environment, records a
    # deferred runtime assertion that value equals hint, keyed by the latest
    # unbacked symbol in the equation, skipping duplicates. Mutates the shape
    # environment (assert list, counter, version). No-op for plain ints or when
    # there is no shape environment.
    if not isinstance(value, torch.SymInt):
        return
    shape_env = getattr(value.node, "shape_env", None)
    if shape_env is None:
        return
    import sympy

    from torch.fx.experimental.symbolic_shapes import (
        free_unbacked_symbols,
        RuntimeAssert,
    )
    from torch.utils._traceback import CapturedTraceback

    expr = sympy.Eq(value.node.expr, hint)
    cands = sorted(
        free_unbacked_symbols(expr),
        key=lambda symbol: int(symbol.name[1:]),
    )
    key = cands[-1] if cands else None
    runtime_asserts = shape_env.deferred_runtime_asserts.setdefault(key, [])
    if any(runtime_assert.expr == expr for runtime_assert in runtime_asserts):
        return
    runtime_asserts.append(
        RuntimeAssert(
            expr,
            f"Runtime assertion failed: symbolic {reason} {value.node.expr} == {hint}",
            CapturedTraceback.extract(skip=1),
        )
    )
    shape_env.num_deferred_runtime_asserts += 1
    shape_env._update_version_counter()


def _split_factor_key(value: IntLikeType) -> object:
    # [FUNCTION DOC] HELPER: returns a hashable, comparable key for a split factor:
    # the symbolic expression for symbolic ints, otherwise the plain int.
    if isinstance(value, torch.SymInt):
        return value.node.expr
    return int(value)


def _sym_mod(left: IntLikeType, right: IntLikeType) -> Any:
    # [FUNCTION DOC] HELPER: modulo of left by right, typed loosely so that it
    # works for ints and symbolic ints.
    return cast(Any, left) % cast(Any, right)


def _hint_proves_even_shard(size: IntLikeType, num_chunks: IntLikeType) -> bool:
    # [FUNCTION DOC] HELPER: returns True if size is known to divide evenly by
    # num_chunks, either provably without guarding or via explicit hints (then it
    # also adds a check that the remainder is zero). Returns False when
    # divisibility cannot be established (missing hints, zero chunk hint, or a
    # nonzero remainder).
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if guard_or_false(_sym_mod(size, num_chunks) == 0):
        return True
    size_hint = _explicit_or_backed_hint(size)
    chunk_hint = _explicit_or_backed_hint(num_chunks)
    if (
        size_hint is None
        or chunk_hint is None
        or chunk_hint == 0
        or size_hint % chunk_hint != 0
    ):
        return False

    # Padding decisions require a Python branch. Only explicit hint overrides
    # can justify skipping padding for otherwise data-dependent sharding.
    torch._check(_sym_mod(size, num_chunks) == 0)
    return True


class Shard(torch._C._distributed.Shard):
    # [CLASS DOC]
    # Placement for a DTensor dimension sharded over one mesh dimension, with the
    # tensor dimension index held in the dim attribute (from the C++ base). Uses
    # chunk semantics: when the size is not divisible, trailing shards are smaller
    # or empty, and collectives operate on padded equal-size shards that are
    # unpadded afterwards. Provides splitting, scatter, reduce-scatter,
    # gather-to-replicate, and shard-dim-change (all-to-all) helpers.
    """
    The ``Shard(dim)`` placement describes the DTensor sharding on tensor dimension
    ``dim`` over a corresponding ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension only holds a shard/piece of the global Tensor. The
    ``Shard(dim)`` placement follows the ``torch.chunk(dim)`` semantic, where the
    last few shards on the DeviceMesh dimension might be empty when the tensor dimension
    is not evenly divisible on the DeviceMesh dimension. The ``Shard`` placement can be
    used by all DTensor APIs (i.e. distribute_tensor, from_local, etc.)

    Args:
        dim (int): The tensor dimension that describes the DTensor is sharded over its
            corresponding DeviceMesh dimension.

    .. warning:: sharding on a tensor dimension where the tensor dimension size is not
        evenly divisible on a DeviceMesh dimension is currently experimental and subject to change.

    .. note:: When checking whether a placement is shard-like, use
        :func:`_is_shard_like` instead of ``isinstance(p, Shard)`` to also
        match :class:`_StridedShard`.
    """

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> tuple[list[torch.Tensor], list[int]]:
        # [FUNCTION DOC]
        # Splits tensor into num_chunks shards along this placement's dim and
        # returns (shards, pad_sizes), delegating to the static helper. With
        # padding, shards are padded to equal size and pad_sizes lists the pad per
        # shard; otherwise pad_sizes is empty. contiguous makes shards contiguous.
        """
        This function uses torch.chunk to split a tensor into num_chunks shards along
        the Shard placement dimension, and return a list of shards with their pad sizes.

        Keyword args:
            with_padding (bool, optional): when True, we pad the tensor on the last
            few ranks before calling the collectives (i.e. scatter/all_gather, etc.).
            This is because collectives usually require equal size tensor inputs
        """
        return self._split_tensor_helper(
            tensor, num_chunks, with_padding, contiguous, self.dim
        )

    @staticmethod
    def _split_tensor_helper(
        tensor: torch.Tensor,
        num_chunks: int,
        with_padding: bool,
        contiguous: bool,
        dim: int,
    ) -> tuple[list[torch.Tensor], list[int]]:
        # [FUNCTION DOC] HELPER: chunks tensor into num_chunks pieces along dim (via
        # the custom chunk routine, which fills missing trailing chunks with empty
        # tensors), optionally pads each to the full chunk size and records the
        # pad amounts, and optionally makes each contiguous. Returns the shard list
        # and pad size list. Raises AssertionError on an out-of-range dim.
        # KNOWN ISSUE: the range check uses greater-than, so dim equal to the tensor
        # rank is not rejected here (it would fail later in the chunk call).
        if dim > tensor.ndim:
            raise AssertionError(
                f"Sharding dim {dim} greater than tensor ndim {tensor.ndim}"
            )

        # chunk tensor over dimension `dim` into n slices
        tensor_list = Shard._custom_chunk(tensor, num_chunks, dim=dim)

        # compute the chunk size inline with ``torch.chunk`` to calculate padding
        full_chunk_size = (tensor.size(dim) + num_chunks - 1) // num_chunks

        shard_list: list[torch.Tensor] = []
        pad_sizes: list[int] = []
        for shard in tensor_list:
            if with_padding:
                pad_size = Shard._get_shard_pad_size(full_chunk_size, shard, dim)
                shard = pad_tensor(shard, dim, pad_size)
                pad_sizes.append(pad_size)
            if contiguous:
                shard = shard.contiguous()
            shard_list.append(shard)
        return shard_list, pad_sizes

    @maybe_run_for_local_tensor
    def _select_split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        index: RankType,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
        clone: bool = True,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns the single shard at index of tensor split into num_chunks along
        # dim, optionally padded, then cloned or made contiguous. For a symbolic
        # index without padding it computes just that slice directly using a
        # narrow with symbolic start and length (the last chunk takes the
        # remainder). Raises AssertionError if dim is out of range. Wrapped to run
        # per local tensor for local-tensor mode.
        """
        Like _split_tensor() but only returns a single shard at the given index.

        This function splits a tensor into num_chunks shards along the Shard placement
        dimension and returns only the shard at the specified index.

        Keyword args:
            with_padding (bool, optional): when True, we pad the tensor on the last
                few ranks before calling the collectives (i.e. scatter/all_gather, etc.).
                This is because collectives usually require equal size tensor inputs.
            contiguous (bool, optional): when True, the returned shard is made contiguous.
            clone (bool, optional): when True, the returned shard is cloned.
        """
        # We don't handle SymInt with_padding yet (because that requires extra
        # work based on the shard)
        if isinstance(index, int) or with_padding:
            shards, _ = self._split_tensor(
                tensor, num_chunks, with_padding=with_padding, contiguous=False
            )
            result = shards[index]
            if clone:
                result = result.clone()
            elif contiguous:
                result = result.contiguous()
            return result

        # For the SymInt implementation just compute the value for the tensor we
        # want rather than computing all of them.

        if self.dim > tensor.ndim:
            raise AssertionError(
                f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"
            )

        # chunk tensor over dimension `dim` into n slices
        dim_size = tensor.size(self.dim)
        split_size = (dim_size + num_chunks - 1) // num_chunks
        # each split is split_size except (maybe) the last one...
        last_split = dim_size - split_size * (num_chunks - 1)

        start = split_size * index
        length = torch.sym_ite(index == num_chunks - 1, last_split, split_size)
        result = torch.narrow(tensor, self.dim, start, length)
        if clone:
            result = result.clone()
        elif contiguous:
            result = result.contiguous()
        return result

    @staticmethod
    def _custom_chunk(
        tensor: torch.Tensor, num_chunks: IntLikeType, dim: int
    ) -> list[torch.Tensor]:
        # [FUNCTION DOC] HELPER: returns exactly num_chunks chunks of tensor along
        # dim. In eager mode (or without unbacked symbols) uses chunk and pads the
        # list with empty tensors; when tracing with unbacked sizes uses narrow with
        # symbolic bounds (an even-split fast path when divisibility is provable).
        # Raises AssertionError for a zero-dimensional tensor or non-positive chunk
        # count. May register deferred runtime asserts for symbolic counts.
        """
        Returns list of tensor chunks along dim.
        Uses torch.chunk in eager mode, but torch.narrow under tracing to be unbacked-symint safe.
        Also handles uneven/zero-sharding cases.
        """
        from torch.distributed._functional_collectives import _are_we_tracing
        from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

        if tensor.dim() <= 0:
            raise AssertionError(f"Expected tensor.dim() > 0, got {tensor.dim()}")
        num_chunks_int = _guarded_hint_int(num_chunks, reason="chunk count")
        if num_chunks_int <= 0:
            raise AssertionError(f"Expected num_chunks > 0, got {num_chunks}")

        # TODO(pianpwk): remove the unbacked symbols check and fix AsyncTP pattern matching
        # for test_micro_pipeline_tp.py.
        if not _are_we_tracing() or not has_free_unbacked_symbols(tensor):
            tensor_list = list(torch.chunk(tensor, num_chunks_int, dim=dim))
            return fill_empty_tensor_to_shards(
                tensor_list, dim, num_chunks_int - len(tensor_list)
            )
        else:
            dim_size = tensor.size(dim)
            if _hint_proves_even_shard(dim_size, num_chunks):
                chunk_size = dim_size // num_chunks
                return [
                    tensor.narrow(dim, chunk_size * i, chunk_size)
                    for i in range(num_chunks_int)
                ]

            split_size = (dim_size + num_chunks - 1) // num_chunks
            chunks = []
            for i in range(num_chunks_int):
                start = torch.sym_min(split_size * i, dim_size)
                end = torch.sym_min(split_size * (i + 1), dim_size)
                chunks.append(tensor.narrow(dim, start, end - start))
            return chunks

    @staticmethod
    @maybe_run_for_local_tensor
    def local_shard_size_and_offset(
        curr_local_size: IntLikeType,
        num_chunks: int,
        rank: _RankTypeT,
    ) -> tuple[_RankTypeT, _RankTypeT]:
        # [FUNCTION DOC]
        # Static method. Given the current local size and number of chunks, returns
        # (local shard size, offset) for rank, following chunk semantics: when the
        # size is provably divisible, size and offset are the full chunk size and
        # chunk size times rank; otherwise the size is clamped (possibly to zero)
        # and the offset clamped to the size using symbolic min/max. Offset is
        # relative to the current local tensor.
        """
        Given the size of the current local tensor (which may already be sharded on some dimensions),
        computes the new local shard size and offset given the desired number of chunks
        (num_chunks is generally equal to the size of the current sharding dim).

        Note: new local shard offset is relative to the current sharded tensor, not the global tensor.
        See `_utils.compute_local_shape_and_global_offset` for computing global offset.

        Returns (new local shard size, offset)

        """
        from torch.fx.experimental.symbolic_shapes import guard_or_false

        # Compute the chunk size inline with ``torch.chunk``
        # Assume uneven sharding as general case for unbacked sizes.
        if guard_or_false(curr_local_size % num_chunks == 0):
            full_chunk_size = curr_local_size // num_chunks
            # pyrefly: ignore[bad-assignment] # pyrefly bug?
            shard_starting_idx: _RankTypeT = full_chunk_size * rank
            return full_chunk_size, shard_starting_idx  # pyrefly: ignore[bad-return]

        # uneven sharding case
        full_chunk_size = (curr_local_size + num_chunks - 1) // num_chunks
        # pyrefly: ignore[bad-assignment] # pyrefly bug?
        shard_starting_idx: _RankTypeT = full_chunk_size * rank

        shard_end_idx = torch.sym_min(
            curr_local_size, shard_starting_idx + full_chunk_size
        )
        local_shard_size = torch.sym_max(0, shard_end_idx - shard_starting_idx)
        return local_shard_size, torch.sym_min(curr_local_size, shard_starting_idx)  # type: ignore[return-value]

    def _local_shard_size_and_offset(
        self,
        curr_local_size: int,
        num_chunks: int,
        rank: RankType,
    ) -> tuple[int, RankType]:
        # [FUNCTION DOC] HELPER: instance-method wrapper delegating to the static
        # Shard.local_shard_size_and_offset; returns (size, offset).
        # pyrefly: ignore [bad-argument-type, bad-return]
        return Shard.local_shard_size_and_offset(curr_local_size, num_chunks, rank)

    @staticmethod
    @maybe_run_for_local_tensor
    def _maybe_unpad_tensor_with_sizes(
        dim, local_tensor, pad_sizes, mesh_dim_local_rank, make_contiguous
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: removes padding along dim from local_tensor if the
        # pad size recorded for this rank (indexed by mesh_dim_local_rank) is
        # (or may be, for symbolic sizes) positive; optionally makes the result
        # contiguous. Returns the possibly-unpadded tensor.
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        # Only unpad if the local_tensor was padded on the dimension.
        # Assume padding (uneven sharding) as general case for unbacked sizes.
        if guard_or_true(pad_sizes[mesh_dim_local_rank] > 0):
            local_tensor = unpad_tensor(
                local_tensor, dim, pad_sizes[mesh_dim_local_rank]
            )
            if make_contiguous:
                local_tensor = local_tensor.contiguous()
        return local_tensor

    def _shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Produces this rank's local shard of tensor along mesh dimension mesh_dim.
        # If the rank is not in the mesh, returns an empty tensor. If src_data_rank
        # is None, just selects the local split with no communication and no
        # padding. Otherwise splits with padding on every rank, scatters from the
        # group source rank src_data_rank, and unpads the received shard.
        # Raises AssertionError if padded shards differ in shape. Collective call.
        """
        Shard and scatter a tensor on a mesh dimension (use coordinate 0 on the
        mesh dimension as source of truth).

        Create the local tensor for this rank following the given Shard
        placement. If src_data_rank is None, perform only local splitting.
        Otherwise, additionally scatter data from src_data_rank. Unlike
        ``_split_tensor``, which supports uneven sharding via padding, this
        method requires the tensor dimension to be evenly divisible by the
        number of chunks (mesh dimension size).
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        mesh_dim_local_rank = my_coordinate[mesh_dim]

        if src_data_rank is None:
            # src_data_rank specified as None explicitly means to skip the
            # communications, simply split
            return self._select_split_tensor(
                tensor,
                num_chunks,
                mesh_dim_local_rank,
                with_padding=False,
                contiguous=True,
            )

        scatter_list, pad_sizes = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )

        it = iter(scatter_list)
        first = next(it)
        # Tensors in the scatter list are expected to have the same shape because
        # split is requested with padding.
        if not all(first.shape == v.shape for v in it):
            raise AssertionError

        output = torch.empty_like(first)

        # perform scatter from the src_data_rank as data source when it is not None
        mesh_scatter(
            output, scatter_list, mesh, mesh_dim=mesh_dim, group_src=src_data_rank
        )

        return Shard._maybe_unpad_tensor_with_sizes(
            self.dim, output, pad_sizes, mesh_dim_local_rank, True
        )

    @classmethod
    def _make_shard_tensor(
        cls,
        dim: int,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: class-level convenience that builds a placement of
        # this class for dim and shards tensor with it; returns the local shard.
        shard_placement = cls(dim)
        return shard_placement._shard_tensor(tensor, mesh, mesh_dim, src_data_rank)

    def _reduce_shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        reduce_op: str,
        mesh_dim: int,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Reduces tensor across the mesh dimension with reduce_op and scatters the
        # result along this placement's dim (reduce-scatter). Pads the input to
        # equal chunks first if the dim is (or may be) unevenly divisible, then
        # unpads the output for this rank. If the rank is not in the mesh, returns
        # tensor unchanged. Collective call.
        """
        reduce and scatter a tensor on a mesh dimension
        """
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return tensor

        # Assume padding (uneven sharding) as general case for unbacked sizes,
        # unless an optimization hint can prove the even-shard branch.
        is_padded = not _hint_proves_even_shard(
            tensor.size(self.dim), num_chunks
        ) and guard_or_true(tensor.size(self.dim) % num_chunks != 0)
        pad_sizes = None
        if is_padded:
            scattered_list, pad_sizes = self._split_tensor(
                tensor, num_chunks, with_padding=True, contiguous=True
            )
            tensor = torch.cat(scattered_list, dim=self.dim)
        elif not tensor.is_contiguous():
            tensor = tensor.contiguous()

        output = funcol.reduce_scatter_single(
            tensor, reduce_op, scatter_dim=self.dim, group=(mesh, mesh_dim)
        )

        if is_padded:
            if pad_sizes is None:
                raise AssertionError
            output = Shard._maybe_unpad_tensor_with_sizes(
                self.dim, output, pad_sizes, mesh._sym_get_coordinate(mesh_dim), False
            )
        return output

    @maybe_run_for_local_tensor
    def _maybe_pad_tensor(
        self,
        local_tensor: torch.Tensor,
        logical_dim_size: IntLikeType,
        num_chunks: int,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: if the logical dim size is (or may be) not evenly
        # divisible by num_chunks, pads local_tensor along dim up to the full chunk
        # size; always returns a contiguous tensor.
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        # Assume padding (uneven sharding) as general case for unbacked sizes,
        # unless an optimization hint can prove the even-shard branch.
        is_padded = not _hint_proves_even_shard(
            logical_dim_size, num_chunks
        ) and guard_or_true(logical_dim_size % num_chunks != 0)

        if is_padded:
            full_chunk_size = (logical_dim_size + num_chunks - 1) // num_chunks
            pad_size = full_chunk_size - local_tensor.size(self.dim)
            local_tensor = pad_tensor(local_tensor, self.dim, pad_size)  # type: ignore[bad-argument-type]

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        return local_tensor

    @maybe_run_for_local_tensor
    def _maybe_unpad_tensor(
        self,
        local_tensor: torch.Tensor,
        logical_dim_size: IntLikeType,
        num_chunks: int,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: after a gather of padded shards, removes trailing
        # padding along dim when padding may have been applied, then narrows the
        # dim to exactly logical_dim_size (rebinding derived symbolic sizes to the
        # original symbol, with a check that the size is large enough). Always
        # returns a contiguous tensor. Raises RuntimeError if the narrow changed
        # the concrete size unexpectedly.
        from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

        # Assume padding (uneven sharding) as general case for unbacked sizes,
        # unless an optimization hint can prove the even-shard branch.
        is_padded = not _hint_proves_even_shard(
            logical_dim_size, num_chunks
        ) and guard_or_true(logical_dim_size % num_chunks != 0)

        if is_padded:
            full_chunk_size = (logical_dim_size + num_chunks - 1) // num_chunks
            unpad_size = full_chunk_size * num_chunks - logical_dim_size  # type: ignore[possibly-undefined]
            local_tensor = unpad_tensor(local_tensor, self.dim, unpad_size)  # type: ignore[bad-argument-type]

        # Bind derived symbolic sizes (e.g. 2*(s//2)) back to the original
        # symbol - needed for correct shape propagation and dynamo generation
        if local_tensor.size(self.dim) is not logical_dim_size:
            orig_size = local_tensor.size(self.dim)
            torch._check(orig_size >= logical_dim_size)
            local_tensor = local_tensor.narrow(self.dim, 0, logical_dim_size)

            # Safety check: the narrow should never change the concrete size.
            # Use guard_or_false so we don't trigger data-dependent guards
            # on unbacked symints.
            if guard_or_false(local_tensor.size(self.dim) != orig_size):
                raise RuntimeError(
                    f"narrow unexpectedly changed concrete size on dim {self.dim}: "
                    f"{orig_size} -> {local_tensor.size(self.dim)}"
                )

        # narrow/unpad on a non-leading dim leaves the tensor non-contiguous,
        # which leaks out to callers that expect a Replicate() result to have
        # a contiguous local tensor (e.g. downstream view ops in TP linear).
        # Use unconditional .contiguous() to avoid guarding on symbolic strides
        # under torch.compile with unbacked symints.
        local_tensor = local_tensor.contiguous()

        return local_tensor

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: Sequence[IntLikeType],
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Converts a sharded local tensor to the full replicated tensor along this
        # dim: pads the local shard to the full chunk size, all-gathers along dim
        # over (mesh, mesh_dim), then unpads to current_logical_shape at dim.
        # Returns the replicated tensor. Collective call.
        """
        This function all_gather all shards and return a tensor that
        is replicated on the previously sharded mesh dimension
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        logical_dim_size = current_logical_shape[self.dim]

        local_tensor = self._maybe_pad_tensor(
            local_tensor, logical_dim_size, num_chunks
        )

        result = funcol.all_gather_single(
            local_tensor,
            gather_dim=self.dim,
            group=(mesh, mesh_dim),
        )

        result = self._maybe_unpad_tensor(result, logical_dim_size, num_chunks)

        return result

    def _replicate_to_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_index: IntLikeType,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Local (no communication) conversion of a replicated tensor to this rank's
        # shard: selects and clones the chunk at shard_index along dim without
        # padding, with chunk count equal to the mesh dimension size.
        """
        transform from replicated tensor to a sharded tensor on
        the current rank, which would perform a local chunk
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        return self._select_split_tensor(
            local_tensor,
            num_chunks,
            shard_index,
            with_padding=False,
            clone=True,
        )

    def _to_partial_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: Sequence[IntLikeType],
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Converts a local shard into a Partial(sum) local tensor: allocates a
        # zero tensor of the full logical size along dim and adds the shard at
        # this rank's offset via index add. Adds a check that the local size
        # matches the expected shard size. Returns the zero-padded tensor.
        """Embed a local shard in a zero-filled tensor pending sum reduction."""
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        logical_dim_size = current_logical_shape[self.dim]
        local_shard_size, local_offset = self.local_shard_size_and_offset(
            logical_dim_size,
            num_chunks,
            mesh._sym_get_coordinate(mesh_dim),
        )
        torch._check(local_tensor.size(self.dim) == local_shard_size)
        output_shape: list[IntLikeType] = list(local_tensor.shape)
        output_shape[self.dim] = logical_dim_size
        output = local_tensor.new_zeros(output_shape)
        offset = torch.scalar_tensor(
            local_offset, dtype=torch.int64, device=local_tensor.device
        )
        indices = torch.arange(
            local_shard_size, dtype=torch.int64, device=local_tensor.device
        )
        return output.index_add(self.dim, indices + offset, local_tensor)

    @staticmethod
    @maybe_run_for_local_tensor
    def _get_shard_pad_size(
        full_size: int, local_tensor: torch.Tensor, dim: int
    ) -> int:
        # [FUNCTION DOC] HELPER: returns the amount of padding needed along dim to
        # bring local_tensor up to full_size (full_size minus current size).
        """
        Get the padding size of the local tensor on the shard dimension.
        """
        return full_size - local_tensor.size(dim)

    @staticmethod
    def _compute_padding_info(
        current_logical_shape: Sequence[IntLikeType],
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
    ) -> tuple[bool, IntLikeType, int, bool, IntLikeType, int]:
        # [FUNCTION DOC] HELPER: for the old and new shard dims (in that order),
        # computes whether padding is needed (size not divisible by num_chunks,
        # decided with an unbacked-safe guard), the logical size, and the full
        # chunk size. Returns a flat six-tuple: old dim triple then new dim triple.
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        results = []
        for shard_dim in [old_shard_dim, new_shard_dim]:
            dim_logical_size = current_logical_shape[shard_dim]
            dim_padding = guard_or_true(dim_logical_size % num_chunks != 0)
            dim_full_chunk_size = (dim_logical_size + num_chunks - 1) // num_chunks
            results.append((dim_padding, dim_logical_size, dim_full_chunk_size))

        return results[0] + results[1]  # type: ignore[bad-return]

    @staticmethod
    @maybe_run_for_local_tensor
    def _pad_for_new_shard_dim(
        current_logical_shape: Sequence[IntLikeType],
        local_tensor: torch.Tensor,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: prepares a local tensor for an all-to-all that moves
        # sharding from old_shard_dim to new_shard_dim: pads the old dim to its full
        # chunk size and the new dim to chunk size times num_chunks when uneven,
        # and returns a contiguous tensor.
        (
            old_dim_padding,
            _,
            old_dim_full_chunk_size,
            new_dim_padding,
            _,
            new_dim_full_chunk_size,
        ) = Shard._compute_padding_info(
            current_logical_shape, num_chunks, old_shard_dim, new_shard_dim
        )

        if old_dim_padding:
            old_dim_pad_size = Shard._get_shard_pad_size(
                old_dim_full_chunk_size, local_tensor, old_shard_dim
            )
            local_tensor = pad_tensor(local_tensor, old_shard_dim, old_dim_pad_size)
        if new_dim_padding:
            new_dim_pad_size = Shard._get_shard_pad_size(
                new_dim_full_chunk_size * num_chunks, local_tensor, new_shard_dim
            )
            local_tensor = pad_tensor(local_tensor, new_shard_dim, new_dim_pad_size)

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()
        return local_tensor

    @staticmethod
    @maybe_run_for_local_tensor
    def _unpad_for_new_shard_dim(
        current_logical_shape: Sequence[IntLikeType],
        local_tensor: torch.Tensor,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
        local_rank: int,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: undoes the padding after an all-to-all that moved
        # sharding from old_shard_dim to new_shard_dim: removes the old dim padding
        # (full size minus logical size) and trims the new dim down to this rank's
        # local shard size (computed from local_rank). Returns the unpadded tensor.
        (
            old_dim_padding,
            _,
            old_dim_full_chunk_size,
            new_dim_padding,
            new_dim_logical_size,
            new_dim_full_chunk_size,
        ) = Shard._compute_padding_info(
            current_logical_shape, num_chunks, old_shard_dim, new_shard_dim
        )

        if old_dim_padding:
            old_dim_unpad_size = (
                old_dim_full_chunk_size * num_chunks
                - current_logical_shape[old_shard_dim]  # type: ignore[possibly-undefined]
            )
            local_tensor = unpad_tensor(local_tensor, old_shard_dim, old_dim_unpad_size)  # type: ignore[possibly-undefined]

        if new_dim_padding:
            local_shard_size_on_new_dim = Shard.local_shard_size_and_offset(
                new_dim_logical_size, num_chunks, local_rank
            )[0]
            new_dim_unpad_size = new_dim_full_chunk_size - local_shard_size_on_new_dim  # type: ignore[possibly-undefined]
            local_tensor = unpad_tensor(local_tensor, new_shard_dim, new_dim_unpad_size)  # type: ignore[possibly-undefined]

        return local_tensor

    def _to_new_shard_dim(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: Sequence[IntLikeType],
        new_shard_dim: int,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Re-shards the local tensor from this placement's dim to new_shard_dim over
        # the mesh dimension: pads, performs an all-to-all, then unpads. If the rank
        # is not in the mesh, returns local_tensor unchanged. Collective call.
        """
        transform from existing sharded tensor to a new sharded tensor on
        that shard on a new dimension, which performs an alltoall
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return local_tensor

        num_chunks = mesh.size(mesh_dim=mesh_dim)

        local_tensor = Shard._pad_for_new_shard_dim(
            current_logical_shape, local_tensor, num_chunks, self.dim, new_shard_dim
        )

        new_tensor = shard_dim_alltoall(
            local_tensor, self.dim, new_shard_dim, mesh, mesh_dim
        )

        new_tensor = Shard._unpad_for_new_shard_dim(
            current_logical_shape,
            new_tensor,
            num_chunks,
            self.dim,
            new_shard_dim,
            my_coordinate[mesh_dim],
        )

        return new_tensor

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash based on the shard dimension only.
        return hash(self.dim)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the machine-readable text naming Shard and its dim.
        """
        machine readable representation of the Shard placement
        """
        return f"Shard(dim={self.dim})"

    def __fx_repr__(self):
        # [FUNCTION DOC] Returns a pair: source text that reconstructs this Shard
        # (fully-qualified constructor call with dim) and an empty globals mapping.
        """
        Returns FX-evaluable repr and required globals for Shard placement.
        Needed for passing this type as an opaque object input to a custom op.
        """
        return f"torch.distributed.tensor.placement_types.Shard(dim={self.dim})", {}

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the short human-readable form: S followed by dim.
        """human readable representation of the Shard placement"""
        return f"S({self.dim})"


class _StridedShard(torch._C._distributed.StridedShard):
    # [CLASS DOC]
    # Internal placement for right-to-left sharding (e.g. FSDP2 plus TP): the dim is
    # first split into split_factor pieces, each piece is split across the mesh
    # dimension, and each rank concatenates its parts from every piece. Holds dim,
    # a split factor (possibly symbolic; the C++ base stores a concrete int and an
    # optional symbolic value is kept on the Python side). Overrides splitting,
    # gather-to-replicate, offset computation, and padding for this layout.
    """
    _StridedShard was originally introduced to support 2D FSDP2 + TP sharding where the tensor
    is sharded on the TP mesh dimension first, then sharded on the FSDP mesh dimension.
    We call this right-to-left sharding which is the opposite of the default
    left-to-right sharding. See the example below::

        tensor shape: [8, 8]
        mesh: [[0, 1], [2, 3]], names=("dp", "tp")
        placements: [Shard(0), Shard(0)]

    The default sharding behavior shards the tensor on "dp" mesh dimension first then
    "tp" dimension. The sharding result will be::

        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   1 (row 2-3)
        2       |   (1, 0)          |   2 (row 4-5)
        3       |   (1, 1)          |   3 (row 6-7)

    While the FSDP2 + TP sharding behavior does the opposite: it shards the tensor on
    "tp" mesh dim first then "dp" dim. This right-to-left sharding will produce the
    result::

        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   2 (row 4-5)
        2       |   (1, 0)          |   1 (row 2-3)
        3       |   (1, 1)          |   3 (row 6-7)

    The consequence is, any attempt to redistribute this DTensor to a full replica will
    produce a wrong result because the shard-to-replicate redistribution always happens
    right-to-left, regardless it's left-to-right sharding or right-to-left. To address
    this, we use _StridedShard placement to make this right-to-left sharding compatible
    with our left-to-right convention on both tensor distribution and redistribution.

    Now with _StridedShard, the right-to-left sharding above can be represented as::

        tensor shape: [8, 8]
        mesh: [[0, 1], [2, 3]], names=("dp", "tp")
        placements: [_StridedShard(0, split_factor=2), Shard(0)]

    And a left-to-right processing of `placements` will produce the same result, which is
    different from using the `Shard` placement::

        Rank    |   Mesh Coordinate |   Shard Index
        ------------------------------------------------
        0       |   (0, 0)          |   0 (row 0-1)
        1       |   (0, 1)          |   2 (row 4-5)
        2       |   (1, 0)          |   1 (row 2-3)
        3       |   (1, 1)          |   3 (row 6-7)

    The argument `split_factor` is the number of existing shards over the tensor sharding
    dimension before processing the _StridedShard placement, as if the sharding happened
    right-to-left. In the example above, the tensor should first be sharded on the "tp"
    dimension into 2 shards before being sharded on the "dp" dimension. Therefore, the
    `split_factor` of the _StridedShard placement on "dp" dim is 2.
    """

    def __init__(
        self,
        dim: int,
        *,
        split_factor: IntLikeType = 1,
        sf: IntLikeType | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Creates the placement for dim with the given split factor; sf is an alias
        # of split_factor. Raises TypeError if both are given with a non-default
        # split_factor. The factor is specialized to a concrete int for the C++
        # base (adding a deferred equality assert for symbolic values, raising
        # RuntimeError without a hint) and the symbolic value is remembered.
        # The C++ placement stores the unroll count as int64_t, but view
        # propagation may derive the split factor from symbolic shape metadata.
        # Preserve that symbolic value on the Python wrapper and guard the
        # concrete unroll count used by list-based lowering.
        if sf is not None:
            if split_factor != 1:
                raise TypeError("_StridedShard received both split_factor and sf")
            split_factor = sf
        split_factor_int = _guarded_hint_int(
            split_factor, reason="_StridedShard split_factor"
        )
        super().__init__(dim, split_factor=split_factor_int)
        if isinstance(split_factor, torch.SymInt):
            self._symbolic_split_factor = split_factor

    @property
    def split_factor(self) -> IntLikeType:  # pyrefly: ignore [bad-override]
        # [FUNCTION DOC] Property: the symbolic split factor if one was given at
        # construction, otherwise the concrete int stored by the C++ base.
        symbolic_split_factor = getattr(self, "_symbolic_split_factor", None)
        if symbolic_split_factor is not None:
            return symbolic_split_factor
        return int(_STRIDED_SHARD_BASE_SPLIT_FACTOR.__get__(self, type(self)))

    def _split_factor_int(self) -> int:
        # [FUNCTION DOC] HELPER: the concrete integer split factor read from the C++
        # base, ignoring any symbolic value.
        return int(_STRIDED_SHARD_BASE_SPLIT_FACTOR.__get__(self, type(self)))

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a _StridedShard with the same dim and
        # the same split factor (compared by symbolic expression when symbolic).
        return (
            isinstance(other, _StridedShard)
            and self.dim == other.dim
            and _split_factor_key(self.split_factor)
            == _split_factor_key(other.split_factor)
        )

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash combining dim and the split factor key.
        return hash((self.dim, _split_factor_key(self.split_factor)))

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the machine-readable text with dim and split factor.
        """
        machine readable representation of the _StridedShard placement
        """
        return f"_StridedShard(dim={self.dim}, sf={self.split_factor})"

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the short form: _S with dim and split factor.
        """human readable representation of the _StridedShard placement"""
        return f"_S({self.dim}, {self.split_factor})"

    @staticmethod
    @maybe_run_for_local_tensor
    def _select_shard(shards: list[torch.Tensor], shard_index) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: returns a clone of the shard at shard_index.
        return shards[shard_index].clone()

    def __fx_repr__(self):
        # [FUNCTION DOC] Returns a pair: source text that reconstructs this
        # _StridedShard using the concrete integer split factor, and an empty
        # globals mapping.
        # KNOWN ISSUE: the docstring says it is for the Shard placement.
        """
        Returns FX-evaluable repr and required globals for Shard placement.
        Needed for passing this type as an opaque object input to a custom op.
        """
        # FX repr cannot refer to this placement's private SymInt object. The
        # constructor emitted a guard tying the symbolic split factor to this
        # concrete unroll count before the placement can reach codegen.
        return (
            f"torch.distributed.tensor.placement_types._StridedShard(dim={self.dim}, sf={self._split_factor_int()})",
            {},
        )

    @classmethod
    def _make_shard_tensor(
        cls,
        dim: int,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
        split_factor: IntLikeType = 1,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: builds a placement for dim with split_factor and
        # shards tensor with it; returns the local shard.
        strided_shard_placement = cls(dim=dim, split_factor=split_factor)
        return strided_shard_placement._shard_tensor(
            tensor, mesh, mesh_dim, src_data_rank
        )

    def _shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Produces this rank's strided shard of tensor along mesh dimension
        # mesh_dim. Rank outside the mesh gets an empty tensor. With src_data_rank
        # None, selects the local strided split without communication. Otherwise
        # splits with padding, scatters from src_data_rank, and unpads. Raises
        # AssertionError if padded shards differ in shape. Collective call.
        """
        Shard and scatter a tensor on a mesh dimension (use coordinate 0 on the
        mesh dimension as source of truth).

        Create the local tensor for this rank following the given StridedShard
        placement. If src_data_rank is None, perform only local splitting.
        Otherwise, additionally scatter data from src_data_rank. Unlike
        ``_split_tensor``, which supports uneven sharding via padding, this
        method requires the tensor dimension to be evenly divisible by the
        number of chunks (mesh dimension size).
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        mesh_dim_local_rank = my_coordinate[mesh_dim]

        if src_data_rank is None:
            # src_data_rank specified as None explicitly means to skip the
            # communications, simply split
            scatter_list, _ = self._split_tensor(
                tensor, num_chunks, with_padding=False, contiguous=True
            )

            return self._select_shard(scatter_list, mesh_dim_local_rank)

        scatter_list, pad_sizes = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )

        it = iter(scatter_list)
        first = next(it)
        # Tensors in the scatter list are expected to have the same shape because
        # split is requested with padding.
        if not all(first.shape == v.shape for v in it):
            raise AssertionError

        output = torch.empty_like(first)

        # perform scatter from the src_data_rank as data source when it is not None
        mesh_scatter(
            output, scatter_list, mesh, mesh_dim=mesh_dim, group_src=src_data_rank
        )

        return Shard._maybe_unpad_tensor_with_sizes(
            self.dim, output, pad_sizes, mesh_dim_local_rank, True
        )

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> tuple[list[torch.Tensor], list[int]]:
        # [FUNCTION DOC]
        # Splits tensor along dim into num_chunks strided shards: first into
        # split_factor pieces, then each piece into num_chunks, and shard i is the
        # concatenation of the i-th sub-piece of every piece. Returns (shards,
        # pad_sizes); with padding, pad_sizes holds each shard's gap to the largest
        # shard (padding itself is not applied here), otherwise it is empty.
        # Raises AssertionError on out-of-range dim.
        # KNOWN ISSUE: the range check uses greater-than, so dim equal to the tensor
        # rank is not rejected by it.
        if self.dim > tensor.ndim:
            raise AssertionError(
                f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"
            )

        # Essentially _StridedShard express the right-to-left sharding in the
        # reversed order. Here we perform first_split as the virtual "right" sharding,
        # and then second_split as the virtual "left" sharding, and finally assemble
        # results in the transposed left-first order.

        # First split: chunk into split_factor pieces
        split_factor = self.split_factor
        split_factor_int = self._split_factor_int()
        first_split = Shard._custom_chunk(tensor, split_factor, dim=self.dim)

        # Second split: chunk each piece into num_chunks pieces
        second_split = []
        for s in first_split:
            second_split.append(Shard._custom_chunk(s, num_chunks, dim=self.dim))

        shard_list: list[torch.Tensor] = []
        for i in range(num_chunks):
            shard = torch.cat(
                [second_split[j][i] for j in range(split_factor_int)],
                dim=self.dim,
            )
            if contiguous:
                shard = shard.contiguous()
            shard_list.append(shard)

        pad_sizes: list[int] = []
        if with_padding:
            # The amount of padding is determined by the local chunk with the
            # largest size. Avoid this Python max when padding is disabled so
            # tracing paths can keep unbacked sizes symbolic.
            max_chunk_size = max([shard.size(self.dim) for shard in shard_list])
            pad_sizes = [max_chunk_size - shard.size(self.dim) for shard in shard_list]

        return shard_list, pad_sizes

    @maybe_run_for_local_tensor
    def _select_split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        index: IntLikeType,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
        clone: bool = True,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns only the strided shard at index (computes all splits, then picks
        # one), cloned if clone is set, else made contiguous if contiguous is set.
        """
        Like _split_tensor() but only returns a single shard at the given index.

        This function splits a tensor into num_chunks shards along the _StridedShard
        placement dimension and returns only the shard at the specified index.

        Keyword args:
            with_padding (bool, optional): when True, we pad the tensor on the last
                few ranks before calling the collectives (i.e. scatter/all_gather, etc.).
                This is because collectives usually require equal size tensor inputs.
            contiguous (bool, optional): when True, the returned shard is made contiguous.
            clone (bool, optional): when True, the returned shard is cloned.
        """
        shards, _ = self._split_tensor(
            tensor, num_chunks, with_padding=with_padding, contiguous=False
        )
        result = shards[index]
        if clone:
            result = result.clone()
        elif contiguous:
            result = result.contiguous()
        return result

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: Sequence[IntLikeType],
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Reconstructs the replicated tensor from strided shards: builds an index
        # tensor and splits it the same way to learn which original positions each
        # rank holds, pads the local shard to the maximum shard size, all-gathers,
        # and uses an inverse permutation with an index select to drop padding and
        # restore original order. Returns a contiguous tensor with the logical
        # size at dim. Collective call.
        """
        Replay the replicate-to-shard process to understand how to stitch shards back.

        This method performs all_gather to collect all shards and then reconstructs
        the original replicated tensor by handling padding, unpadding, and reordering.

        Example:
            Consider a 1D input tensor [0, 1, 2, 3, 4, 5, 6, 7, 8] with 9 elements.
            Using _StridedShard(dim=0, split_factor=2) and num_chunks=4:

            Preparation (via _split_tensor, before _to_replicate_tensor is called):
                _split_tensor produces 4 shards with strided indices:
                - First split (split_factor=2): [0,1,2,3,4] and [5,6,7,8]
                - Second split (num_chunks=4) on each piece:
                    [0,1,2,3,4] (5 elements) -> [[0,1], [2,3], [4], []]
                    [5,6,7,8]   (4 elements) -> [[5], [6], [7], [8]]
                - Transpose and concatenate for each chunk:
                    Chunk 0: [0,1] + [5] = [0,1,5]  (indices)
                    Chunk 1: [2,3] + [6] = [2,3,6]  (indices)
                    Chunk 2: [4] + [7]   = [4,7]    (indices)
                    Chunk 3: [] + [8]    = [8]      (indices)

                So we get shards with values:
                    Rank 0: [0, 1, 5]  (size=3)
                    Rank 1: [2, 3, 6]  (size=3)
                    Rank 2: [4, 7]     (size=2)
                    Rank 3: [8]        (size=1)

                These shards are the `local_tensor` input to _to_replicate_tensor
                on each rank. Each rank only has its own shard when this function
                is called.

            Step 1: Pad all shards to max_chunk_size=3:
                    Rank 0: [0, 1, 5]     (no padding needed)
                    Rank 1: [2, 3, 6]     (no padding needed)
                    Rank 2: [4, 7, P]     (padded with 1 element)
                    Rank 3: [8, P, P]     (padded with 2 elements)

            Step 2: all_gather produces concatenated padded tensor:
                [0, 1, 5, | 2, 3, 6, | 4, 7, P, | 8, P, P]
                 chunk 0    chunk 1    chunk 2    chunk 3
                (pos 0-2)  (pos 3-5)  (pos 6-8)  (pos 9-11)

            Step 3: Compute select_indices to extract valid elements and reorder:
                sharded_indices = [[0,1,5], [2,3,6], [4,7], [8]]
                padded_positions:
                    chunk 0: base=0 -> [0, 1, 2]  (positions of [0,1,5] in gathered)
                    chunk 1: base=3 -> [3, 4, 5]  (positions of [2,3,6] in gathered)
                    chunk 2: base=6 -> [6, 7]     (positions of [4,7] in gathered)
                    chunk 3: base=9 -> [9]        (position of [8] in gathered)

                permutation = cat(sharded_indices) = [0, 1, 5, 2, 3, 6, 4, 7, 8]
                select_positions = cat(padded_positions) = [0, 1, 2, 3, 4, 5, 6, 7, 9]

                inv_permutation = argsort(permutation)
                    permutation[0]=0 -> inv_permutation[0]=0
                    permutation[1]=1 -> inv_permutation[1]=1
                    permutation[2]=5 -> inv_permutation[5]=2
                    permutation[3]=2 -> inv_permutation[2]=3
                    permutation[4]=3 -> inv_permutation[3]=4
                    permutation[5]=6 -> inv_permutation[6]=5
                    permutation[6]=4 -> inv_permutation[4]=6
                    permutation[7]=7 -> inv_permutation[7]=7
                    permutation[8]=8 -> inv_permutation[8]=8
                    => inv_permutation = [0, 1, 3, 4, 6, 2, 5, 7, 8]

                select_indices = select_positions[inv_permutation]
                               = [0, 1, 2, 3, 4, 5, 6, 7, 9][inv_permutation]
                    For original position 0: select_indices[0] = select_positions[0] = 0
                    For original position 1: select_indices[1] = select_positions[1] = 1
                    For original position 2: select_indices[2] = select_positions[3] = 3
                    For original position 3: select_indices[3] = select_positions[4] = 4
                    For original position 4: select_indices[4] = select_positions[6] = 6
                    For original position 5: select_indices[5] = select_positions[2] = 2
                    For original position 6: select_indices[6] = select_positions[5] = 5
                    For original position 7: select_indices[7] = select_positions[7] = 7
                    For original position 8: select_indices[8] = select_positions[8] = 9
                    => select_indices = [0, 1, 3, 4, 6, 2, 5, 7, 9]

            Step 4: index_select from gathered tensor using select_indices:
                gathered = [0, 1, 5, 2, 3, 6, 4, 7, P, 8, P, P]
                result = gathered[select_indices]
                       = gathered[[0, 1, 3, 4, 6, 2, 5, 7, 9]]
                       = [0, 1, 2, 3, 4, 5, 6, 7, 8]

            The result is the original replicated tensor [0, 1, 2, 3, 4, 5, 6, 7, 8].
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        logical_dim_size = current_logical_shape[self.dim]

        # indices_tensor is 1D torch.arange(logical_dim_size) unsqueezed
        # so that we can reuse self._split_tensor which splits on self.dim
        shape = [1] * self.dim + [logical_dim_size]
        # pyrefly: ignore [no-matching-overload]
        indices_tensor = torch.arange(
            logical_dim_size, device=local_tensor.device
        ).view(shape)

        sharded_indices, _ = self._split_tensor(
            indices_tensor,
            num_chunks,
            with_padding=False,
            contiguous=False,
        )
        # squeeze back to 1D indices tensor
        sharded_indices = [shard.view(-1) for shard in sharded_indices]

        # First chunk should be one of those biggest chunks. Compute the size
        # from the logical shape rather than the fake arange split so padding
        # depends only on the original symbolic dimension and its hint.
        max_chunk_size, _ = self.local_shard_size_and_offset(
            logical_dim_size,
            num_chunks,
            rank=0,
            offset_mode=_StridedShardOffsetMode.ALL,
        )
        local_pad_size = max_chunk_size - local_tensor.size(self.dim)
        local_tensor_padded = pad_tensor(local_tensor, self.dim, local_pad_size)

        if not local_tensor_padded.is_contiguous():
            local_tensor_padded = local_tensor_padded.contiguous()

        replicate_tensor_permuted_padded = funcol.all_gather_single(
            local_tensor_padded,
            gather_dim=self.dim,
            group=(mesh, mesh_dim),
        )
        if isinstance(replicate_tensor_permuted_padded, funcol.AsyncCollectiveTensor):
            replicate_tensor_permuted_padded = replicate_tensor_permuted_padded.wait()

        # After all_gather, the tensor is [chunk0, chunk1 (may padded), ...].
        # Each chunk may have padding at the end. Use a single index_select to
        # both extract non-padding data and reorder to the original positions.
        #
        # Build select_indices where select_indices[original_pos] = position in
        # the padded tensor that holds the element for original_pos.
        padded_positions: list[torch.Tensor] = []
        for i, shard in enumerate(sharded_indices):
            base_offset = i * max_chunk_size
            positions = cast(
                torch.Tensor,
                base_offset
                + torch.arange(
                    shard.numel(),
                    device=local_tensor.device,
                ),
            )
            padded_positions.append(positions)

        # Permutation ends up containing strided indices because we create it by
        # chunking over a particular dimension of an N-D shaped arange tensor.
        permutation = torch.cat(sharded_indices)
        # Choose the position by skipping padding indices from
        # replicate_tensor_permuted_padded.
        select_positions = torch.cat(padded_positions)

        inv_permutation = torch.argsort(permutation)
        select_indices = select_positions.index_select(0, inv_permutation)

        replicate_tensor = torch.index_select(
            replicate_tensor_permuted_padded, self.dim, select_indices
        )
        logical_shape = cast(list[IntLikeType], list(replicate_tensor.shape))
        logical_shape[self.dim] = current_logical_shape[self.dim]
        replicate_tensor = replicate_tensor.as_strided(
            logical_shape,
            torch._prims_common.make_contiguous_strides_for(
                cast(list[int], logical_shape)
            ),
        )

        return replicate_tensor.contiguous()

    def _replicate_to_strided_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_index: IntLikeType,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Local (no communication) conversion of a replicated tensor to this rank's
        # strided shard: selects and clones the strided split at shard_index with
        # chunk count equal to the mesh dimension size, without padding.
        """
        Transform from replicated tensor to a strided-sharded tensor on the current rank.

        This performs a local chunking operation using the _StridedShard pattern,
        where the tensor is split according to the strided sharding semantics
        (interleaved pieces based on split_factor).

        Args:
            local_tensor: The replicated tensor on this rank.
            mesh: The device mesh over which the tensor is distributed.
            mesh_dim: The mesh dimension for the sharding.
            shard_index: The index of the shard to select (typically the rank's
                coordinate on the mesh dimension).

        Returns:
            The local strided shard for this rank.
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        return self._select_split_tensor(
            local_tensor,
            num_chunks,
            shard_index,
            with_padding=False,
            clone=True,
        )

    @staticmethod
    @maybe_run_for_local_tensor
    def _local_shard_size(sharded_indices: list[torch.Tensor], rank: RankType) -> int:
        # [FUNCTION DOC] HELPER: number of elements in the index list belonging to
        # rank.
        return len(sharded_indices[rank])

    def _local_shard_size_and_offset(
        self,
        curr_local_size: int,
        num_chunks: int,
        rank: RankType,
        offset_mode: _StridedShardOffsetMode = _StridedShardOffsetMode.FIRST,
    ) -> tuple[int, int | list[int] | None]:
        # [FUNCTION DOC] HELPER: instance-method wrapper delegating to
        # local_shard_size_and_offset with the same arguments; returns (size, offset).
        return self.local_shard_size_and_offset(
            curr_local_size, num_chunks, rank, offset_mode
        )

    @maybe_run_for_local_tensor
    def local_shard_size_and_offset(
        self,
        curr_local_size: IntLikeType,
        num_chunks: int,
        rank: RankType,
        offset_mode: _StridedShardOffsetMode = _StridedShardOffsetMode.FIRST,
    ) -> tuple[int, list[int] | int | None]:
        # [FUNCTION DOC]
        # Returns (local shard size, offsets) for rank under strided sharding of a
        # dimension of size curr_local_size across num_chunks. Offsets depend on
        # offset_mode: a single first index (minus one if empty) for FIRST, the list
        # of all original indices for ALL, or None for NONE. For symbolic sizes with
        # unbacked symbols and a non-FIRST mode it computes the size symbolically
        # (no offsets list). Otherwise it splits an index tensor and reads the
        # indices of rank. Wrapped to run per local tensor.
        """
        Compute the local shard size and offset(s) for a _StridedShard placement.

        Unlike the regular Shard placement which produces contiguous offsets, _StridedShard
        produces non-contiguous (strided) offsets due to the right-to-left sharding semantics.
        This method computes the actual indices that belong to the local shard.

        Args:
            self (_StridedShard): The _StridedShard placement instance.
            curr_local_size (int): The current size of the tensor dimension to be sharded.
            num_chunks (int): Number of chunks to split the dimension into (typically the mesh dimension size).
            rank (RankType): The rank index to compute the shard for.
            offset_mode (_StridedShardOffsetMode): Controls offset materialization.
                FIRST returns only the first offset as an int, ALL returns all offsets
                as a list, and NONE skips offset materialization. Defaults to FIRST.

        Returns:
            tuple: A tuple containing:
                - local_shard_size (int): The number of elements in the local shard
                  for this rank.
                - offset (int | list[int] | None): Determined by offset_mode. FIRST
                  returns the first offset as an int, ALL returns a list of all offsets,
                  and NONE returns None.
        """
        offset_mode = _StridedShardOffsetMode(offset_mode)
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

        if (
            offset_mode is not _StridedShardOffsetMode.FIRST
            and isinstance(curr_local_size, torch.SymInt)
            and free_unbacked_symbols(curr_local_size.node.expr)
        ):
            split_factor = self.split_factor
            split_factor_int = self._split_factor_int()
            offset = None if offset_mode is _StridedShardOffsetMode.NONE else []
            if _hint_proves_even_shard(curr_local_size, split_factor * num_chunks):
                # pyrefly: ignore[bad-return]
                return (
                    curr_local_size // num_chunks,
                    offset,
                )

            local_shard_size: IntLikeType = 0
            first_split_size = (curr_local_size + split_factor - 1) // split_factor
            for split_idx in range(split_factor_int):
                first_start = torch.sym_min(
                    first_split_size * split_idx, curr_local_size
                )
                first_end = torch.sym_min(
                    first_split_size * (split_idx + 1), curr_local_size
                )
                first_len = torch.sym_max(0, first_end - first_start)
                second_split_size = (first_len + num_chunks - 1) // num_chunks
                second_start = torch.sym_min(second_split_size * rank, first_len)
                second_end = torch.sym_min(second_split_size * (rank + 1), first_len)
                local_shard_size = local_shard_size + torch.sym_max(
                    0, second_end - second_start
                )
            return local_shard_size, offset  # pyrefly: ignore[bad-return]

        # indices_tensor is 1D torch.arange(logical_dim_size) unsqueezed
        # so that we can reuse self._split_tensor which splits on self.dim
        shape = [1] * self.dim + [curr_local_size]
        # pyrefly: ignore [no-matching-overload]
        indices_tensor = torch.arange(
            curr_local_size,
        ).view(shape)

        sharded_indices, _ = self._split_tensor(
            indices_tensor,
            num_chunks,
            with_padding=False,
            contiguous=False,
        )
        # squeeze back to 1D indices tensor
        sharded_indices = [shard.view(-1) for shard in sharded_indices]

        local_shard_size = _StridedShard._local_shard_size(sharded_indices, rank)
        if offset_mode is _StridedShardOffsetMode.NONE:
            # Avoid .tolist() which creates unbacked SymInts under FakeTensorMode.
            return local_shard_size, None
        if local_shard_size > 0:
            offsets = sharded_indices[rank].tolist()
        else:
            offsets = []

        if offset_mode is _StridedShardOffsetMode.FIRST:
            # Always return an int for consistency across ranks.
            # For empty shards, return -1 as an invalid offset indicator.
            offsets = offsets[0] if len(offsets) > 0 else -1

        return local_shard_size, offsets

    @staticmethod
    def _compute_padding_info(
        logical_size_on_dim: int,
        num_chunks: int,
        shard_dim: int,
        split_factor: int = 1,
    ) -> tuple[bool, int]:
        # [FUNCTION DOC] HELPER: static method returning (needs_padding, max_chunk_size)
        # for a dimension of the given logical size split by split_factor and then
        # num_chunks. With split_factor one it is plain ceiling-division chunking;
        # otherwise padding is needed if there is a partial or empty first-level
        # chunk or an uneven second-level split, and max_chunk_size sums the
        # per-level ceilings. The shard_dim argument is unused.
        """
        Compute padding information for _StridedShard collective operations.

        This method calculates whether padding is needed and the maximum chunk size
        for collective operations (e.g., all-to-all) involving _StridedShard tensors.

        For _StridedShard with split_factor > 1, the tensor undergoes two-level splitting:
        1. First level: split into ``split_factor`` pieces
        2. Second level: each piece is split into ``num_chunks`` pieces

        The resulting shards are interleaved, so padding must account for both levels.
        Unlike ``Shard._compute_padding_info`` where at most one partition (the last)
        requires padding, _StridedShard can have multiple trailing partitions that
        need padding due to the interleaved structure.

        When split_factor=1, this behaves identically to regular Shard padding logic.

        Args:
            logical_size_on_dim: The logical shape size of the tensor ``shard_dim``.
            num_chunks: Number of chunks to split into (typically the mesh dim size).
            shard_dim: The dimension along which the tensor is sharded.
            split_factor: The number of pre-existing splits from right-to-left sharding.
                Defaults to 1 (no strided sharding effect).

        Returns:
            A tuple of (needs_padding_on_dim, max_chunk_size):
                - needs_padding_on_dim: Whether padding is required on the shard dimension.
                - max_chunk_size: The maximum chunk size per rank after both levels of splitting.
        """

        def _ceil_div(a: int, b: int) -> int:
            # [FUNCTION DOC] HELPER: integer ceiling division of a by b.
            return (a + b - 1) // b

        if split_factor != 1:
            # Computing padding info for StridedShard tensor dim
            # First level: split into split_factor pieces
            first_chunk_size = _ceil_div(logical_size_on_dim, split_factor)
            num_full_first_chunks = logical_size_on_dim // first_chunk_size
            remainder = logical_size_on_dim - num_full_first_chunks * first_chunk_size

            # Determine if padding is needed:
            # - remainder > 0 means one chunk has partial size
            # - empty first-level chunks means some ranks get nothing at each level
            # - first_chunk_size % num_chunks != 0 means uneven second-level split
            #   (e.g., 6 rows, sf=2, 8 ranks: first_chunk=3, but only 3 of 8 ranks
            #   get data per first-level chunk, so ranks 3-7 have size-0 tensors)
            has_partial_chunk = remainder > 0
            num_empty_chunks = (
                split_factor - num_full_first_chunks - (1 if has_partial_chunk else 0)
            )
            has_uneven_second_level = first_chunk_size % num_chunks != 0
            needs_padding_on_dim = (
                has_partial_chunk or num_empty_chunks > 0 or has_uneven_second_level
            )

            # Second level: each first-level chunk is split into num_chunks pieces
            # Calculate the per-rank chunk size after both levels of splitting
            max_chunk_size = (
                _ceil_div(first_chunk_size, num_chunks) * num_full_first_chunks
            )
            if has_partial_chunk:
                max_chunk_size += _ceil_div(remainder, num_chunks)
        else:
            # Compute padding info for normal shard, no split_factor impact.
            needs_padding_on_dim = logical_size_on_dim % num_chunks != 0
            max_chunk_size = _ceil_div(logical_size_on_dim, num_chunks)

        return (
            needs_padding_on_dim,
            max_chunk_size,
        )

    @maybe_run_for_local_tensor
    def _pad_for_new_shard_dim(
        self,
        current_logical_shape: list[int],
        local_tensor: torch.Tensor,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
        split_factor: int,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: pads local_tensor before the all-to-all in a shard
        # dim change: the old (strided) dim up to its maximum chunk size and the new
        # dim up to maximum chunk size times num_chunks, only when padding is needed
        # and positive. Returns a contiguous tensor.
        """
        Pad the local tensor before alltoall for _StridedShard's _to_new_shard_dim operation.

        For _StridedShard, the padding must account for the split_factor since each rank
        holds split_factor interleaved pieces. The full chunk size per rank is
        split_factor * single_chunk_size.
        """
        (
            old_dim_padding,
            old_dim_max_chunk_size,
        ) = _StridedShard._compute_padding_info(
            current_logical_shape[old_shard_dim],
            num_chunks,
            old_shard_dim,
            split_factor,
        )
        (
            new_dim_padding,
            new_dim_max_chunk_size,
        ) = _StridedShard._compute_padding_info(
            current_logical_shape[new_shard_dim],
            num_chunks,
            new_shard_dim,
        )

        if old_dim_padding:
            target_size = old_dim_max_chunk_size
            old_dim_pad_size = target_size - local_tensor.size(old_shard_dim)
            if old_dim_pad_size > 0:
                local_tensor = pad_tensor(local_tensor, old_shard_dim, old_dim_pad_size)

        if new_dim_padding:
            target_total_size = new_dim_max_chunk_size * num_chunks
            new_dim_pad_size = target_total_size - local_tensor.size(new_shard_dim)
            if new_dim_pad_size > 0:
                local_tensor = pad_tensor(local_tensor, new_shard_dim, new_dim_pad_size)

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()
        return local_tensor

    @maybe_run_for_local_tensor
    def _unpad_for_new_shard_dim(
        self,
        current_logical_shape: list[int],
        local_tensor: torch.Tensor,
        num_chunks: int,
        old_shard_dim: int,
        new_shard_dim: int,
        split_factor: int,
        local_rank: int,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: after the all-to-all, selects the valid elements of
        # the old dim (dropping padding and restoring the original order via an
        # inverse permutation index select), then trims the new dim to this rank's
        # local size when it was padded. Returns the resulting tensor.
        # KNOWN ISSUE: split_factor is accepted but not used in this method.
        """
        Unpad the local tensor after alltoall for _StridedShard's
        _to_new_shard_dim operation.

        This uses the same index_select approach as _to_replicate_tensor to both
        extract correct elements (removing padding) and reorder them. The key insight
        is that padding can only be at the end, so the first shard has the max size.
        """
        old_dim_logical_size = current_logical_shape[old_shard_dim]
        new_dim_logical_size = current_logical_shape[new_shard_dim]
        (
            new_dim_padding,
            new_dim_max_chunk_size,
        ) = _StridedShard._compute_padding_info(
            current_logical_shape[new_shard_dim],
            num_chunks,
            new_shard_dim,
        )

        # Build sharded indices to understand the strided pattern
        shape = [1] * old_shard_dim + [old_dim_logical_size]
        indices_tensor = torch.arange(
            old_dim_logical_size, device=local_tensor.device
        ).view(shape)
        sharded_indices, _ = self._split_tensor(
            indices_tensor, num_chunks, with_padding=False, contiguous=False
        )
        sharded_indices = [shard.view(-1) for shard in sharded_indices]

        max_chunk_size = len(sharded_indices[0])

        # Compute positions in the padded local_tensor for each actual element.
        # Each shard i starts at position i * max_chunk_size in local_tensor.
        padded_positions = []
        for i, shard in enumerate(sharded_indices):
            base_offset = i * max_chunk_size
            positions = base_offset + torch.arange(
                len(shard), device=local_tensor.device
            )
            padded_positions.append(positions)

        permutation = torch.cat(sharded_indices)
        select_positions = torch.cat(padded_positions)

        inv_permutation = torch.argsort(permutation)
        select_indices = select_positions.index_select(0, inv_permutation)
        local_tensor = torch.index_select(local_tensor, old_shard_dim, select_indices)

        if new_dim_padding:
            # copy from Shard._unpad_for_new_shard_dim
            local_shard_size_on_new_dim = Shard.local_shard_size_and_offset(
                new_dim_logical_size, num_chunks, local_rank
            )[0]
            new_dim_unpad_size = new_dim_max_chunk_size - local_shard_size_on_new_dim  # type: ignore[possibly-undefined]
            local_tensor = unpad_tensor(local_tensor, new_shard_dim, new_dim_unpad_size)  # type: ignore[possibly-undefined]
        return local_tensor

    def _to_new_shard_dim(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
        new_shard_dim: int,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Re-shards the local tensor from this strided dim to new_shard_dim over the
        # mesh dimension: pads (using the concrete split factor), all-to-all, then
        # unpads. Rank outside the mesh returns local_tensor unchanged. Collective.
        """
        Transform from existing _StridedShard tensor to a new _StridedShard tensor that
        shards on a new dimension, which performs an alltoall.

        This is analogous to Shard._to_new_shard_dim but accounts for the split_factor
        in padding calculations since _StridedShard represents interleaved sharding.
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return local_tensor

        num_chunks = mesh.size(mesh_dim=mesh_dim)

        local_tensor = self._pad_for_new_shard_dim(
            current_logical_shape,
            local_tensor,
            num_chunks,
            self.dim,
            new_shard_dim,
            self._split_factor_int(),
        )

        new_tensor = shard_dim_alltoall(
            local_tensor, self.dim, new_shard_dim, mesh, mesh_dim
        )
        new_tensor = self._unpad_for_new_shard_dim(
            current_logical_shape,
            new_tensor,
            num_chunks,
            self.dim,
            new_shard_dim,
            self._split_factor_int(),
            my_coordinate[mesh_dim],
        )

        return new_tensor


def _is_shard_like(p: "Placement") -> TypeGuard[Shard | _StridedShard]:
    # [FUNCTION DOC] HELPER: type-narrowing check returning True if p is a Shard or
    # a _StridedShard.
    """Check if a placement is Shard or _StridedShard.

    Use this instead of ``isinstance(p, Shard)`` to avoid silently missing
    ``_StridedShard``.  When ``_StridedShard`` is unified with ``Shard``
    (see TODO on the class), this helper can be collapsed to a single
    ``isinstance`` check.
    """
    return isinstance(p, Shard | _StridedShard)


class Replicate(torch._C._distributed.Replicate):
    # [CLASS DOC]
    # Placement meaning every rank on the mesh dimension holds the full tensor.
    # Stateless: all instances are equal and hash identically. Provides broadcast
    # of the source rank's data to the rest of the mesh dimension.
    """
    The ``Replicate()`` placement describes the DTensor replicating on a corresponding
    ``DeviceMesh`` dimension, where each rank on the DeviceMesh dimension holds a
    replica of the global Tensor. The ``Replicate`` placement can be used by all
    DTensor APIs (i.e. ``distribute_tensor``, ``DTensor.from_local``, etc.)
    """

    def __hash__(self) -> int:
        # [FUNCTION DOC] Constant hash (minus one) since all instances are the same.
        # every replicate placement is the same
        return -1

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the fixed text naming the Replicate placement.
        """
        machine readable representation of the Replicate placement
        """
        return "Replicate()"

    def __fx_repr__(self):
        # [FUNCTION DOC] Returns a pair: fully-qualified constructor source text for
        # Replicate and an empty globals mapping.
        """
        Returns FX-evaluable repr and required globals for Replicate placement.
        Needed for passing this type as an opaque object input to a custom op.
        """
        return "torch.distributed.tensor.placement_types.Replicate()", {}

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the short human-readable form, the letter R.
        """
        human readable representation of the Replicate placement
        """
        return "R"

    @classmethod
    def _make_replicate_tensor(
        cls,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Class-level helper: makes tensor replicated along mesh dimension mesh_dim.
        # A rank outside the mesh gets an empty tensor. Otherwise makes the tensor
        # contiguous and, unless src_data_rank is None, broadcasts in place from
        # the group source rank src_data_rank. Returns the tensor. Collective call.
        """
        Replicate (broadcast) a torch.Tensor on a mesh dimension (use
        the first coordinate on the mesh dimension as source of truth)
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        tensor = tensor.contiguous()

        if src_data_rank is not None:
            # perform broadcast from the src_data_rank as data source when it is not None
            mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim, group_src=src_data_rank)
        return tensor

    def _replicate_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: instance-method wrapper delegating to the class-level
        # replicate helper; returns the replicated tensor.
        return Replicate._make_replicate_tensor(tensor, mesh, mesh_dim, src_data_rank)


class Partial(torch._C._distributed.Partial):
    # [CLASS DOC]
    # Placement meaning each rank holds a partial value that still needs to be
    # reduced with reduce_op (sum by default) across the mesh dimension. Exposes
    # the class constants LINEAR_REDUCE_OPS and ALL_REDUCE_OPS and implements the
    # contract methods: reduce to replicate (all-reduce), reduce to shard
    # (reduce-scatter), and partitioning a replicated value into partial values.
    # KNOWN ISSUE: the long descriptive string placed after the class constants is
    # not the class docstring (it is not the first statement), so it is not
    # exposed as documentation.
    # reduce_ops that distribute over addition, enabling per-input linearity
    # for bilinear ops like mm: reduce_op(A_i @ B) = reduce_op(A_i) @ B
    LINEAR_REDUCE_OPS: tuple[str, ...] = ("sum", "avg")
    ALL_REDUCE_OPS: tuple[str, ...] = ("sum", "avg", "min", "max", "product")

    """
    The ``Partial(reduce_op)`` placement describes the DTensor that is pending
    reduction on a specified ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension holds the partial value of the global Tensor. User can
    redistribute the ``Partial`` DTensor to a ``Replicate`` or ``Shard(dim)``
    placement on the specified ``DeviceMesh`` dimension using ``redistribute``,
    which would trigger necessary communication operations under the hood (i.e.
    ``allreduce``, ``reduce_scatter``).

    Args:
        reduce_op (str, optional): The reduction op to be used for the partial DTensor
            to produce Replicated/Sharded DTensor. Corresponds to the reduce operations
            supported by ``torch.distributed.ReduceOp``. Default: "sum".

            Supported values:

            * ``"sum"``: Element-wise sum across all ranks.
            * ``"avg"``: Element-wise average across all ranks.
            * ``"min"``: Element-wise minimum across all ranks.
            * ``"max"``: Element-wise maximum across all ranks.
            * ``"product"``: Element-wise product across all ranks.
            * ``"band"``: Bitwise AND across all ranks (integer tensors only).
            * ``"bor"``: Bitwise OR across all ranks (integer tensors only).
            * ``"bxor"``: Bitwise XOR across all ranks (integer tensors only).

    .. note:: The ``Partial`` placement can be generated as a result of DTensor
        operators, by ``DTensor.from_local``, or by redistributing a ``Shard``
        placement to ``Partial("sum")``.
    """

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Resolves the partial value to a replicated one: all-reduces tensor with
        # this placement's reduce_op over (mesh, mesh_dim) and returns the result.
        # Collective call.
        # Partial placement contract #1:
        # _reduce_value: reduce the value of the tensor on the mesh dimension
        return funcol.all_reduce(
            tensor, reduceOp=self.reduce_op, group=(mesh, mesh_dim)
        )

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Resolves the partial value to a sharded one: delegates to the given Shard
        # placement's reduce-scatter with this reduce_op and returns this rank's
        # reduced shard. Collective call.
        # Partial placement contract #2:
        # _reduce_shard_value: reduce_scatter the value of the tensor over the mesh dimension
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Turns a replicated tensor into per-rank partial values (conjugate of the
        # reduce): for sum, divides by the mesh dimension size; for avg, min and
        # max, returns the tensor as is. Raises ValueError for any other reduce_op.
        # No communication.
        """
        Partition a replicated tensor to create partial values for Replicate → Partial.

        This is the conjugate operation of _reduce_value. The partition operation
        must satisfy the invariant that applying _reduce_value to the partitioned
        values recovers the original replicated value (modulo floating-point error).

        Mathematical analysis by reduce_op:

        * "sum": partition(v) = v / n, then sum([v/n] * n) = v
          Introduces floating-point error from the division and summation.
          Error grows with n (the number of ranks).

        * "avg": partition(v) = v, then avg([v] * n) = v
          Numerically exact (averaging identical values).

        * "min": partition(v) = v, then min([v] * n) = v
          Numerically exact.

        * "max": partition(v) = v, then max([v] * n) = v
          Numerically exact.

        * "product": Would need partition(v) = v^(1/n), but n-th root is not exact
          for general values and undefined for negative values with even n.
          NOT SUPPORTED.

        * "band"/"bor"/"bxor": Bitwise operations have no well-defined inverse
          that partitions a value such that reducing recovers the original.
          NOT SUPPORTED.
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        if self.reduce_op == "sum":
            return tensor / num_chunks
        elif self.reduce_op in ("avg", "min", "max"):
            return tensor
        else:
            raise ValueError(
                f"Replicate to Partial({self.reduce_op}) conversion is not supported."
            )

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash of the reduce_op, offset by one.
        return 1 + hash(self.reduce_op)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the machine-readable text naming Partial and its
        # reduce_op.
        """
        machine readable representation of the Partial placement
        """
        return f"Partial({self.reduce_op})"

    def __fx_repr__(self):
        # [FUNCTION DOC] Returns a pair: constructor source text for Partial with the
        # quoted reduce_op, and an empty globals mapping.
        """
        Returns FX-evaluable repr and required globals for Partial placement.
        Needed for passing this type as an input to a custom op.
        """
        return (
            f"torch.distributed.tensor.placement_types.Partial({self.reduce_op!r})",
            {},
        )

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the short form: P with the reduce_op.
        """
        human readable representation of the Partial placement
        """
        return f"P({self.reduce_op})"


# We keep the old _Partial name for a while for BC reason
_Partial = Partial


@dataclass(frozen=True)
class _MaskPartial(Partial):
    # [CLASS DOC]
    # Internal frozen-dataclass Partial placement for row-wise sharded embedding:
    # when partitioning the (index) input it records which entries fall outside the
    # local embedding shard in a MaskBuffer and shifts indices into local range;
    # when reducing, it zeroes the masked entries before the reduction. State:
    # mask_buffer (shared mutable buffer whose lifetime follows the DTensor),
    # offset_shape and offset_dim (which full-size dimension defines the shard
    # offsets). Equality compares the buffer by identity.
    """
    A partial mask placement devised for rowwise sharded embedding op, where we need
    to mask and adjust the indices to the local embedding shard, embedding masking
    is a special type of the Partial placement

    NOTE: the lifecycle of this _MaskPartial placement follows the corresponding DTensor
    lifecycle, i.e. the indices_mask would only be alive during the lifetime of the DTensor.
    """

    mask_buffer: MaskBuffer = field(default_factory=MaskBuffer)

    # required fields for computing the local offset and deriving the mask
    offset_shape: torch.Size | None = None
    offset_dim: int = 0

    def __init__(
        self,
        reduce_op=None,
        mask_buffer=None,
        offset_shape=None,
        offset_dim=0,
        *args,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Initializes the Partial base with reduce_op, then sets (bypassing frozen
        # dataclass protection) mask_buffer (a new MaskBuffer if None), offset_shape
        # and offset_dim. Extra positional and keyword arguments are accepted and
        # ignored.
        super().__init__(reduce_op)
        if mask_buffer is None:
            mask_buffer = MaskBuffer()
        object.__setattr__(self, "mask_buffer", mask_buffer)
        object.__setattr__(self, "offset_shape", offset_shape)
        object.__setattr__(self, "offset_dim", offset_dim)

    @staticmethod
    @maybe_run_for_local_tensor
    def _mask_tensor(
        tensor: torch.Tensor,
        local_offset_on_dim: IntLikeType,
        local_shard_size: IntLikeType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: returns (mask, masked_tensor) where mask marks
        # entries of tensor outside the half-open range starting at
        # local_offset_on_dim of length local_shard_size, and masked_tensor is a
        # copy shifted by the offset with masked entries set to zero. Does not
        # modify the input.
        # Build the input mask and save it for the current partial placement
        # this is so that the output of embedding op can reuse the same partial
        # placement saved mask to perform mask + reduction
        mask = (tensor < local_offset_on_dim) | (
            tensor >= local_offset_on_dim + local_shard_size
        )
        # mask the input tensor
        # pyrefly: ignore [unsupported-operation]
        masked_tensor = tensor.clone() - local_offset_on_dim
        masked_tensor[mask] = 0
        # pyrefly: ignore [bad-return]
        return mask, masked_tensor

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Overrides the parent: computes this rank's embedding shard size and offset
        # from offset_shape at offset_dim, masks/shifts the input tensor, stores the
        # mask in mask_buffer (side effect), and returns the masked tensor.
        # Raises AssertionError if the rank is not in the mesh or offset_shape is
        # unset.
        if not mesh._is_current_rank_part_of_mesh():
            raise AssertionError("rank is not part of mesh")
        # override parent logic to perform partial mask for embedding
        num_chunks = mesh.size(mesh_dim)
        # get local shard size and offset on the embedding_dim
        if self.offset_shape is None:
            raise AssertionError("offset_shape needs to be set for _MaskPartial")
        local_shard_size, local_offset_on_dim = Shard.local_shard_size_and_offset(
            self.offset_shape[self.offset_dim],
            num_chunks,
            mesh._sym_get_coordinate(mesh_dim),
        )
        mask, masked_tensor = _MaskPartial._mask_tensor(
            tensor, local_offset_on_dim, local_shard_size
        )
        # materialize the mask buffer to be used for reduction
        self.mask_buffer.materialize_mask(mask)
        return masked_tensor

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Applies the saved mask to tensor in place (zeroing masked entries),
        # releases the mask buffer, and all-reduces with reduce_op over (mesh,
        # mesh_dim). Raises AssertionError if no mask was saved. Collective call.
        # KNOWN ISSUE: the inline comment says sum reduction, but reduce_op is used.
        # by the time we need reduction, we should have already saved the mask
        if self.mask_buffer.data is None:
            raise AssertionError

        # apply the mask to the tensor that pending reduction
        self.mask_buffer.apply_mask(tensor)

        # clear the mask buffer
        self.mask_buffer.release_mask()

        # perform sum reduction
        return funcol.all_reduce(
            tensor, reduceOp=self.reduce_op, group=(mesh, mesh_dim)
        )

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Applies the saved mask to tensor in place, releases the mask buffer, and
        # delegates to the given Shard placement's reduce-scatter with reduce_op.
        # Raises AssertionError if no mask was saved. Collective call.
        # by the time we need reduction, we should have already saved the mask
        if self.mask_buffer.data is None:
            raise AssertionError

        # apply the mask to the tensor that pending reduction
        self.mask_buffer.apply_mask(tensor)

        # clear the mask buffer
        self.mask_buffer.release_mask()

        # call reduce_shard_tensor of the shard_spec.
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other is a _MaskPartial with the same reduce_op,
        # offset_shape, offset_dim and the very same mask buffer object.
        if not isinstance(other, _MaskPartial):
            return False

        return (
            self.reduce_op == other.reduce_op
            and self.offset_shape == other.offset_shape
            and self.offset_dim == other.offset_dim
            and self.mask_buffer is other.mask_buffer
        )

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash of reduce_op, offset_shape, offset_dim and the identity
        # of the mask buffer, offset by one.
        return 1 + hash(
            (
                self.reduce_op,
                self.offset_shape,
                self.offset_dim,
                id(self.mask_buffer),
            )
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the machine-readable text with reduce_op,
        # offset_shape and offset_dim (the mask buffer is omitted).
        """
        machine readable representation of the _MaskPartial placement
        """
        return f"_MaskPartial(reduce_op={self.reduce_op}, offset_shape={self.offset_shape}, offset_dim={self.offset_dim})"

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the short form: MaskP with reduce_op, offset_shape
        # and offset_dim.
        """
        human readable representation of the _MaskPartial placement
        """
        return f"MaskP({self.reduce_op}, {self.offset_shape}, {self.offset_dim})"

    def __fx_repr__(self):
        # [FUNCTION DOC] Returns a pair: constructor source text and an empty globals
        # mapping.
        # KNOWN ISSUE: the text names MaskPartial, but the class is _MaskPartial (no
        # public MaskPartial exists in this module), and reduce_op is not quoted,
        # so the text cannot be evaluated to rebuild the placement.
        """
        Returns FX-evaluable repr and required globals for Partial placement.
        Needed for passing this type as an input to a custom op.
        """
        return (
            f"torch.distributed.tensor.placement_types.MaskPartial(reduce_op={self.reduce_op}, offset_shape={self.offset_shape}, offset_dim={self.offset_dim})",
            {},
        )


def _register_placements_as_opaque():
    # [FUNCTION DOC] HELPER: registers Placement, Shard, Replicate, Partial,
    # _StridedShard and _MaskPartial as constant opaque custom classes, allowing
    # the real implementations of the shard/partial/replicate predicates and
    # equality, plus dim for the shard types and reduce_op for the partial types.
    # Side effect: mutates the global opaque-object registry. Called at import.
    from torch._library.opaque_object import MemberType, register_custom_class

    allowed_members = {
        "is_shard": MemberType.USE_REAL,
        "is_partial": MemberType.USE_REAL,
        "is_replicate": MemberType.USE_REAL,
        "__eq__": MemberType.USE_REAL,
    }
    register_custom_class(Placement, typ="constant", members=allowed_members)
    register_custom_class(
        Shard, typ="constant", members=allowed_members | {"dim": MemberType.USE_REAL}
    )
    register_custom_class(Replicate, typ="constant", members=allowed_members)
    register_custom_class(
        Partial,
        typ="constant",
        members=allowed_members | {"reduce_op": MemberType.USE_REAL},
    )
    register_custom_class(
        _StridedShard,
        typ="constant",
        members=allowed_members | {"dim": MemberType.USE_REAL},
    )
    register_custom_class(
        _MaskPartial,
        typ="constant",
        members=allowed_members | {"reduce_op": MemberType.USE_REAL},
    )


_register_placements_as_opaque()
