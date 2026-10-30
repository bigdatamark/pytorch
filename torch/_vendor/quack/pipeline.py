# Copyright (c) 2025-2026, Tri Dao.
# [MODULE DOC] Extensions of the CUTLASS DSL pipeline classes for quack GEMM kernels. Adds: a
# pipeline state that can advance by many iterations, a state factory, a mixin giving every
# pipeline index/phase based acquire, commit, wait and release variants, a named barrier with
# indexed arrive, pipeline variants with optional elect-one signalling or an extra transaction
# byte count, an mbarrier array that can drop part of the arrive count, and a TMA plus
# cp.async to UMMA pipeline. Each subclass keeps a create factory that builds the original
# CUTLASS object and then switches its class, since the dataclasses are frozen. Import-time
# side effect: assigns the create factory attribute on several of these classes.

from typing import Optional
from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import if_generate, and_, dsl_user_op
from cutlass.pipeline import MbarrierArray, CooperativeGroup, PipelineOp
from cutlass.pipeline import PipelineState, PipelineUserType
from cutlass.pipeline import Agent, agent_sync
from cutlass.pipeline import NamedBarrier as NamedBarrierOg
from cutlass.pipeline import PipelineAsync as PipelineAsyncOg
from cutlass.pipeline import PipelineCpAsync as PipelineCpAsyncOg
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg
from cutlass.pipeline import PipelineTmaUmma as PipelineTmaUmmaOg
from cutlass.pipeline import PipelineUmmaAsync as PipelineUmmaAsyncOg
from cutlass.pipeline import PipelineAsyncUmma as PipelineAsyncUmmaOg


# ── Shared helpers ───────────────────────────────────────────────────────────


def _override_create(parent_cls, child_cls):
    """Create a static factory that constructs parent_cls then re-classes to child_cls."""
    # [FUNCTION DOC] HELPER: Build a static factory method for a subclass. The returned
    # function calls the parent class create with all arguments, then forcibly changes the
    # resulting object's class to the child class (using the low-level attribute setter
    # because the dataclass is frozen), and returns it. Used so child pipelines reuse
    # parent construction.

    @staticmethod
    def create(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Static factory produced by the override helper: creates the
        # parent object with the given arguments and re-classes it to the child class. Returns
        # the object.
        obj = parent_cls.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        object.__setattr__(obj, "__class__", child_cls)
        return obj

    return create


def _make_state(index: Int32, phase: Int32) -> PipelineState:
    """Construct a PipelineState from index and phase (count/stages unused by callers)."""
    # [FUNCTION DOC] HELPER: Make a pipeline state from an index and phase, with the stage
    # count and iteration count set to zero (callers only use index and phase).
    return PipelineState(stages=0, count=Int32(0), index=index, phase=phase)


def _call_with_elect_one(parent_method, self, state, elect_one, syncwarp, loc, ip):
    """Optionally wrap a parent pipeline method call in sync_warp + elect_one."""
    # [FUNCTION DOC] HELPER: Call a parent pipeline method (commit or release) on a state,
    # with optional single-thread signalling. If elect_one is set (compile-time), then
    # optionally (when syncwarp is set) synchronize the warp and call the parent method only
    # inside an elected-thread region; otherwise call it directly from all threads. loc and ip
    # are the location and insertion point passed through to the parent method.
    if const_expr(elect_one):
        if const_expr(syncwarp):
            cute.arch.sync_warp()
        with cute.arch.elect_one():
            parent_method(self, state, loc=loc, ip=ip)
    else:
        parent_method(self, state, loc=loc, ip=ip)


# ── Pipeline state ──────────────────────────────────────────────────────────


class PipelineStateWAdvance(PipelineState):
    # [CLASS DOC] PipelineState with the ability to advance by an arbitrary number of
    # iterations. Same state as the base class (stage count, iteration count, index, phase);
    # the only additions are advance_iters and a rebuild hook that keeps this subclass type.
    @dsl_user_op
    def advance_iters(self, num_iterations: Int32, *, loc=None, ip=None):
        # [FUNCTION DOC] Advance the state by num_iterations steps at once. Adds to the
        # iteration count and the index, flips the phase using the number of times the index
        # wrapped past the stage count, and reduces the index modulo the stage count. Mutates
        # the state in place.
        # KNOWN ISSUE: phase is combined with the raw wrap count rather than its parity, so
        # advancing across two or more full stage cycles would leave the phase outside 0
        # and 1.
        self._count += Int32(num_iterations)
        new_index = self._index + Int32(num_iterations)
        # How many times did we cross the stages boundary
        num_crossings = new_index // self.stages
        self._phase ^= num_crossings
        self._index = new_index % self.stages

    # This can be overridden by derived classes
    def __new_from_mlir_values__(self, values):
        # [FUNCTION DOC] Rebuild the state from MLIR values (iteration count, index, phase, in
        # that order) keeping the current stage count; returns a PipelineStateWAdvance.
        return PipelineStateWAdvance(
            self.stages, Int32(values[0]), Int32(values[1]), Int32(values[2])
        )


def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    # [FUNCTION DOC] Create a PipelineStateWAdvance for a producer or consumer with the given
    # stage count. Both start at iteration zero and index zero; a producer starts with phase 1
    # (so its first acquires pass on an empty buffer) and a consumer with phase 0. Any other
    # user type trips an assertion failure.
    Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
    """
    if type is PipelineUserType.Producer:
        return PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(1))
    elif type is PipelineUserType.Consumer:
        return PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0))
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."


# ── Mixin: _w_index / _w_index_phase variants ───────────────────────────────


class _PipelineIndexPhaseMixin:
    # [CLASS DOC] HELPER: Mixin for pipeline classes whose base methods take a pipeline state.
    # It adds variants taking an explicit stage index (and phase where needed) instead of a
    # state object; each builds a temporary state and delegates to the matching base method,
    # so subclass overrides of those base methods are honored. Holds no state of its own.
    """Mixin providing _w_index_phase / _w_index methods that delegate to PipelineState-based parents."""

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        # [FUNCTION DOC] Producer acquire given a stage index and phase instead of a
        # state. Builds a temporary state and calls producer_acquire with the optional
        # try-acquire token.
        state = _make_state(index, phase)
        self.producer_acquire(state, try_acquire_token, loc=loc, ip=ip)

    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None):
        # [FUNCTION DOC] Producer commit given only a stage index. Builds a temporary state
        # (phase zero, which commit does not use) and calls producer_commit.
        state = _make_state(index, Int32(0))
        self.producer_commit(state, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        # [FUNCTION DOC] Consumer wait given a stage index and phase. Builds a temporary state
        # and calls consumer_wait with the optional try-wait token.
        state = _make_state(index, phase)
        self.consumer_wait(state, try_wait_token, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        # [FUNCTION DOC] Consumer release given only a stage index. Builds a temporary state
        # (phase zero) and calls consumer_release.
        state = _make_state(index, Int32(0))
        self.consumer_release(state, loc=loc, ip=ip)


# ── NamedBarrier ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NamedBarrier(NamedBarrierOg):
    # [CLASS DOC] Frozen-dataclass named barrier extending the CUTLASS one with arrive and
    # wait operations on a barrier id offset by a runtime index, letting one object cover
    # several consecutive hardware barrier ids. The create factory is replaced right after the
    # class definition so it returns this subclass.
    create = _override_create(NamedBarrierOg, None)  # patched below

    @dsl_user_op
    def arrive_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        # [FUNCTION DOC] Arrive without waiting on the barrier whose id is the base id plus
        # index, expecting the configured number of threads. Returns nothing.
        """
        The aligned flavor of arrive is used when all threads in the CTA will execute the
        same instruction. See PTX documentation.
        """
        cute.arch.barrier_arrive(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def arrive_and_wait_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        # [FUNCTION DOC] Arrive and wait on the barrier whose id is the base id plus index,
        # for the configured number of threads. Returns nothing.
        cute.arch.barrier(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )


NamedBarrier.create = _override_create(NamedBarrierOg, NamedBarrier)


# ── PipelineAsync ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineAsync(_PipelineIndexPhaseMixin, PipelineAsyncOg):
    # [CLASS DOC] Frozen-dataclass async pipeline with optional single-thread signalling.
    # Four private fields (set by create after construction) choose whether commit and
    # release use an elected thread per warp and whether to synchronize the warp first.
    # Overrides producer_commit and consumer_release accordingly; the index based variants
    # come from the mixin.
    """
    PipelineAsync with optional elect_one for producer_commit and consumer_release.

    When elect_one_*=True (set at create time), only one elected thread per warp
    signals the barrier arrive. This is useful when the mask count is set to 1 per warp.

    Args (to create):
        elect_one_commit: If True, only elected thread signals producer_commit.
        syncwarp_before_commit: If True (default), issue syncwarp before elect_one.
        elect_one_release: If True, only elected thread signals consumer_release.
        syncwarp_before_release: If True (default), issue syncwarp before elect_one.
            Set syncwarp to False when threads are already converged (e.g. after wgmma wait_group).
    """

    _elect_one_commit: bool = False
    _syncwarp_before_commit: bool = True
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        elect_one_commit: bool = False,
        syncwarp_before_commit: bool = True,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        # [FUNCTION DOC] Create a PipelineAsync from the base create arguments plus four
        # flags: elect_one_commit, syncwarp_before_commit, elect_one_release,
        # syncwarp_before_release. Builds the base object, switches its class and stores
        # the flags using the low-level setter (the dataclass is frozen). Returns the
        # new pipeline.
        obj = PipelineAsyncOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineAsync)
        object.__setattr__(obj, "_elect_one_commit", elect_one_commit)
        object.__setattr__(obj, "_syncwarp_before_commit", syncwarp_before_commit)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def producer_commit(self, state: PipelineState, *, loc=None, ip=None):
        # [FUNCTION DOC] Producer commit that optionally runs the base commit from only one
        # elected thread, according to the commit flags, with an optional warp sync first.
        _call_with_elect_one(
            PipelineAsyncOg.producer_commit,
            self,
            state,
            self._elect_one_commit,
            self._syncwarp_before_commit,
            loc,
            ip,
        )

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        # [FUNCTION DOC] Consumer release that optionally runs the base release from only one
        # elected thread, according to the release flags, with an optional warp sync first.
        _call_with_elect_one(
            PipelineAsyncOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )

    # _w_index variants inherited from _PipelineIndexPhaseMixin, which delegate
    # to producer_commit / consumer_release above.


# ── PipelineCpAsync ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineCpAsync(_PipelineIndexPhaseMixin, PipelineCpAsyncOg):
    # [CLASS DOC] Frozen-dataclass cp.async pipeline with optional single-thread consumer
    # release. Two private fields (set by create) choose the elected-thread behavior and
    # whether to sync the warp first; only consumer_release is overridden. Index based
    # variants come from the mixin.
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        # [FUNCTION DOC] Create a PipelineCpAsync from the base create arguments plus the
        # flags elect_one_release and syncwarp_before_release. Builds the base object,
        # switches its class and stores the flags with the low-level setter. Returns the
        # new pipeline.
        obj = PipelineCpAsyncOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineCpAsync)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        # [FUNCTION DOC] Consumer release that optionally runs the base release from only one
        # elected thread, per the release flags, with an optional warp sync first.
        _call_with_elect_one(
            PipelineCpAsyncOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )

    # _w_index variants inherited from _PipelineIndexPhaseMixin.


# ── PipelineTmaAsync ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaAsync(_PipelineIndexPhaseMixin, PipelineTmaAsyncOg):
    # [CLASS DOC] TMA async pipeline whose producer_acquire accepts an extra transaction
    # byte count. No new state; the create factory is reassigned after the class so it
    # returns this type.
    """Override producer_acquire to take in extra_tx_count parameter."""

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        # [FUNCTION DOC] Producer acquire for TMA loads. Waits on the empty barrier unless a
        # nonzero try-acquire token says the buffer is already known to be free. Then arrives
        # on the full barrier of the stage: a plain arrive with the producer mask when
        # extra_tx_count is zero (checked at compile time), otherwise an arrive that also
        # expects the full barrier's transaction bytes plus extra_tx_count. Returns nothing.
        # KNOWN ISSUE: the existing docstring calls this a producer commit, but it is
        # an acquire.
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip)
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(state.index, tx_count, loc=loc, ip=ip)


PipelineTmaAsync.create = _override_create(PipelineTmaAsyncOg, PipelineTmaAsync)


# ── PipelineTmaUmma ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaUmma(_PipelineIndexPhaseMixin, PipelineTmaUmmaOg):
    # [CLASS DOC] TMA-to-UMMA pipeline whose producer_acquire accepts an extra transaction
    # byte count and an is_tma_warp flag. No new state; the create factory is reassigned after
    # the class.
    """Override producer_acquire to take in extra_tx_count parameter."""

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        is_tma_warp: Optional[Boolean] = True,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        # [FUNCTION DOC] Producer acquire for TMA loads feeding UMMA. Waits on the empty
        # barrier unless a nonzero try-acquire token is given. Only a thread in the
        # leader CTA that is also in the TMA warp arrives on the full barrier, since
        # several warps may call this: a plain arrive with the producer mask when
        # extra_tx_count is zero, otherwise an arrive that also expects the full
        # barrier's transaction bytes plus extra_tx_count.
        # KNOWN ISSUE: the existing docstring calls this a producer commit, but it is
        # an acquire.
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        # This is the difference between this and PipelineTmaAsync: we could have multiple
        # warps calling this, but only 1 warp should do the arrive on the full barrier
        if const_expr(extra_tx_count == 0):
            if_generate(
                and_(self.is_leader_cta, is_tma_warp),
                lambda: self.sync_object_full.arrive(
                    state.index, self.producer_mask, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            if_generate(
                and_(self.is_leader_cta, is_tma_warp),
                lambda: self.sync_object_full.arrive_and_expect_tx(
                    state.index, tx_count, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )


PipelineTmaUmma.create = _override_create(PipelineTmaUmmaOg, PipelineTmaUmma)


# ── PipelineUmmaAsync ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineUmmaAsync(_PipelineIndexPhaseMixin, PipelineUmmaAsyncOg):
    # [CLASS DOC] UMMA async pipeline with the index based variants from the mixin; no other
    # changes. The create factory is reassigned after the class so it returns this type.
    pass


PipelineUmmaAsync.create = _override_create(PipelineUmmaAsyncOg, PipelineUmmaAsync)


# ── PipelineAsyncUmma ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineAsyncUmma(_PipelineIndexPhaseMixin, PipelineAsyncUmmaOg):
    # [CLASS DOC] Async-to-UMMA pipeline with the index based variants from the mixin; no
    # other changes. The create factory is reassigned after the class so it returns this type.
    pass


PipelineAsyncUmma.create = _override_create(PipelineAsyncUmmaOg, PipelineAsyncUmma)


# ── PipelineTmaCpAsync ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaCpAsync(_PipelineIndexPhaseMixin, PipelineTmaAsyncOg):
    # [CLASS DOC] TMA async pipeline for mixed cp.async and TMA producers with async-thread
    # consumers. Differs from the TMA async pipeline in that producer_acquire only arrives on
    # the full barrier from the TMA warp, and in having producer_cpasync_commit. Create is
    # reassigned after the class to return this type.
    """
    PipelineTmaCpAsync is used for CpAsync + TMA producers and AsyncThread consumers.
    Compared to PipelineTmaAsync, producer_acquire gates the full-barrier arrive on is_tma_warp.
    """

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        is_tma_warp: Optional[Boolean] = True,
        *,
        loc=None,
        ip=None,
    ):
        # [FUNCTION DOC] Producer acquire: waits on the empty barrier unless a nonzero try-
        # acquire token is given, then arrives on the full barrier with the producer mask
        # only when is_tma_warp is true, because several warps may call this but only one
        # should arrive.
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        # This is the difference between this and PipelineTmaAsync: we could have multiple
        # warps calling this, but only 1 warp should do the arrive on the full barrier
        if_generate(
            is_tma_warp,
            lambda: self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_cpasync_commit(self, state: PipelineState, *, loc=None, ip=None):
        # [FUNCTION DOC] Make the full barrier of the given stage track completion of the
        # thread's earlier cp.async copies, by an arrive-on-completion that does not
        # increment the pending count. Uses the barrier returned by the producer barrier
        # lookup for the state.
        """We need the mbarrier to track the completion of cp.async."""
        cute.arch.cp_async_mbarrier_arrive_noinc(
            self.producer_get_barrier(state, loc=loc, ip=ip), loc=loc, ip=ip
        )


PipelineTmaCpAsync.create = _override_create(PipelineTmaAsyncOg, PipelineTmaCpAsync)


# ── MbarrierArrayWDropCount ─────────────────────────────────────────────────


class MbarrierArrayWDropCount(MbarrierArray):
    # [CLASS DOC] MbarrierArray variant whose arrive count can be reduced by a runtime drop
    # count. State: barrier storage pointer, transaction byte count, stage count, operation
    # type and cooperative group, arrive count (group size minus the drop count if given) and
    # drop count. Initializes the mbarriers in the constructor. Only the storage pointer and
    # drop count are exposed as MLIR values.
    @dsl_user_op
    def __init__(
        self,
        barrier_storage: cute.Pointer,
        num_stages: int,
        agent: tuple[PipelineOp, CooperativeGroup],
        tx_count: int = 0,
        drop_count: Optional[Int32] = None,
        *,
        loc=None,
        ip=None,
    ) -> None:
        # [FUNCTION DOC] Store the configuration, validate it and initialize the mbarriers.
        # Args: shared-memory barrier storage, stage count, an agent pair of operation type
        # and group, transaction bytes (default 0) and optional drop count subtracted from the
        # group size to get the arrive count. Raises ValueError for a non-positive stage
        # count, a non-positive group size, or a negative tx count with the TMA load
        # operation. The group size check happens before the drop count is subtracted. Side
        # effect: emits mbarrier initialization.
        self.barrier_storage = barrier_storage
        self.tx_count = tx_count
        self.num_stages = num_stages
        self.op_type, self.cg = agent
        self.arrive_count = self.cg.size
        self.drop_count = drop_count

        if self.num_stages <= 0:
            raise ValueError("Error: Mbarrier stage count must be greater than 0.")
        if self.arrive_count <= 0:
            raise ValueError("Error: Mbarrier arrive count must be greater than 0.")
        if self.op_type is PipelineOp.TmaLoad and self.tx_count < 0:
            raise ValueError("Error: Mbarrier tx count must not be less than 0 for TMA ops.")

        if const_expr(drop_count is not None):
            self.arrive_count = self.arrive_count - drop_count

        # Store mbarrier base pointer
        self.mbarrier_base = self.barrier_storage

        # Mbarrier initialization in constructor
        self.mbarrier_init(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        # [FUNCTION DOC] Return the values that cross MLIR boundaries: the barrier storage
        # pointer and the drop count.
        return [self.barrier_storage, self.drop_count]

    def __new_from_mlir_values__(self, values):
        # [FUNCTION DOC] Rebuild an instance from MLIR values (storage pointer, drop count),
        # reusing the stage count, agent, and transaction count of this instance.
        return MbarrierArrayWDropCount(
            values[0], self.num_stages, (self.op_type, self.cg), self.tx_count, values[1]
        )


# ── PipelineTmaCpAsyncUmma ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaCpAsyncUmma(PipelineTmaUmmaOg):
    # [CLASS DOC] Frozen-dataclass TMA pipeline for Blackwell mainloops where A/B are loaded
    # by a mix of cp.async and TMA producers and consumed by UMMA. Extends the TMA-to-UMMA
    # base with a custom create (using a drop-count full mbarrier array), a producer_acquire
    # gated on leader CTA and TMA warp, and producer_cpasync_commit.
    """
    PipelineTmaCpAsync is used for CpAsync + TMA producers and UMMA consumers
    (e.g. Blackwell mainloops)
    """

    @dsl_user_op
    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        mcast_mode_mn: tuple[int, int] = (1, 1),
        defer_sync: bool = False,
        producer_drop_count: Optional[Int32] = None,
        loc=None,
        ip=None,
    ):
        # [FUNCTION DOC] Create the pipeline (keyword-only args): stage count, producer and
        # consumer cooperative groups, tx_count, barrier storage pointer, optional cluster
        # layout, multicast modes (default 1,1), defer_sync, and optional producer_drop_count.
        # Raises TypeError if barrier storage is not a pointer. The full barrier array is the
        # drop-count variant on the aligned storage; the empty barrier array follows it, built
        # for the TCGen05 MMA consumer. With no cluster layout or a single CTA there is no
        # multicast mask and the CTA is leader; otherwise the mask and leader flag are
        # computed. The CTA group is one or two depending on the first cluster mode. The
        # consumer mask equals the producer mask. Unless defer_sync is set, issues an mbarrier
        # init fence and a thread-block (or relaxed cluster) sync.
        # KNOWN ISSUE: the existing docstring says it raises ValueError and describes a
        # different class name; it also omits the drop count and defer_sync parameters.
        """Creates and initializes a new PipelineTmaUmma instance.

        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: int
        :param producer_group: CooperativeGroup for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: CooperativeGroup for the consumer agent
        :type consumer_group: CooperativeGroup
        :param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx_count: int
        :param barrier_storage: Pointer to the shared memory address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer, optional
        :param cta_layout_vmnk: Layout of the cluster shape
        :type cta_layout_vmnk: cute.Layout, optional
        :param mcast_mode_mn: Tuple specifying multicast modes for m and n dimensions (each 0 or 1)
        :type mcast_mode_mn: tuple[int, int], optional
        :raises ValueError: If barrier_storage is not a cute.Pointer instance
        :return: A new PipelineTmaUmma instance configured with the provided parameters
        :rtype: PipelineTmaUmma
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise TypeError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TmaLoad
        consumer_type = PipelineOp.TCGen05Mma

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = MbarrierArrayWDropCount(
            barrier_storage.align(min_align=8),
            num_stages,
            producer,
            tx_count,
            drop_count=producer_drop_count,
            loc=loc,
            ip=ip,
        )
        sync_object_empty = PipelineTmaUmmaOg._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages,
            num_stages,
            consumer,
            loc=loc,
            ip=ip,
        )

        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, loc=loc, ip=ip) == 1:
            # No mcast mask if not using clusters
            producer_mask = None
            # All threadblocks are leaders if not using clusters
            is_leader_cta = True
        else:
            producer_mask = PipelineTmaUmmaOg._compute_mcast_arrival_mask(
                cta_layout_vmnk, mcast_mode_mn, loc=loc, ip=ip
            )
            is_leader_cta = PipelineTmaUmmaOg._compute_is_leader_cta(
                cta_layout_vmnk, loc=loc, ip=ip
            )

        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0], loc=loc, ip=ip) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )

        consumer_mask = producer_mask

        if not defer_sync:
            cute.arch.mbarrier_init_fence()
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, loc=loc, ip=ip) == 1:
                agent_sync(Agent.ThreadBlock)
            else:
                agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)

        return PipelineTmaCpAsyncUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            is_leader_cta,
            cta_group,
        )

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        is_tma_warp: Optional[Boolean] = True,
        *,
        loc=None,
        ip=None,
    ):
        # [FUNCTION DOC] Producer acquire: waits on the empty barrier unless a nonzero try-
        # acquire token is given, then arrives on the full barrier with the producer mask only
        # when this CTA is the leader and is_tma_warp is true.
        # KNOWN ISSUE: the existing docstring calls this a producer commit, but it is
        # an acquire.
        """
        TMA producer commit conditionally waits on buffer empty and sets the
        transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        # This is the difference between this and PipelineTmaAsync: we could have multiple
        # warps calling this, but only 1 warp should do the arrive on the full barrier
        if_generate(
            and_(self.is_leader_cta, is_tma_warp),
            lambda: self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_cpasync_commit(self, state: PipelineState, *, loc=None, ip=None):
        # [FUNCTION DOC] Make the full barrier of the stage track completion of the thread's
        # earlier cp.async copies, via an arrive-on-completion that does not increment the
        # pending count.
        """
        We need the mbarrier to track the completion of cp.async
        """
        cute.arch.cp_async_mbarrier_arrive_noinc(
            self.producer_get_barrier(state, loc=loc, ip=ip), loc=loc, ip=ip
        )
