# Copyright (c) 2025, Tri Dao.
# [MODULE DOC] ComposableEpiMixin: builds GEMM epilogue hook methods and the EpilogueParams dataclass from a class-level tuple of
# EpiOp objects (_epi_ops), delegating smem sizing, smem layout, TMA atoms and per-tile device hooks to each op. Contains
# _make_epi_params and ComposableEpiMixin. No import-time side effects beyond imports.
"""ComposableEpiMixin: composes EpiOps into epilogue hook methods.

Subclasses declare _epi_ops as a class-level tuple of EpiOp instances — the
static *schema* for the epilogue. The mixin auto-generates epi_smem_bytes,
epi_get_smem_struct, epi_get_smem_tensors, epi_begin, epi_begin_loop,
epi_end_loop, epi_end, and EpilogueParams by querying each op.

Host-side, `_epi_ops_to_params_dict` (called from each subclass's
`epi_to_underlying_arguments`) filters `_epi_ops` automatically: it shadows
the class-level tuple with an instance-level tuple containing only the ops
whose argument tensor is non-None. All later iteration (host- and
device-side) walks the filtered tuple, so each EpiOp's hook methods can
assume their `param`/`arg_tensor` is non-None.

The two host-side hooks that run *before* `epi_to_underlying_arguments`
(`resolve_epi_m_major` and the classmethod `epi_smem_bytes`) filter inline
from `args`, preserving the same non-None invariant for `op.epi_m_major_score`
and `op.smem_bytes`. They have to run first because `epi_to_underlying_arguments`
itself depends on static attributes set up before it (e.g. `gemm.epi_tile`,
`gemm.epi_c_stage`), and those attributes are themselves derived from
`epi_m_major` and the epi smem budget — a chicken-and-egg ordering we resolve by
letting these two hooks see the raw `args` and filter inline.

epi_get_smem_tensors, epi_begin, and epi_begin_loop all return dicts keyed by
op name, so consumers access values by name (e.g. epi_smem_tensors["mAuxOut"],
epi_loop_tensors["alpha"]). Because inactive ops are filtered out, consumers
must use `.get(name)` (returns None for inactive ops) rather than `[name]`.

EpilogueParams is auto-generated from the full class-level _epi_ops (via
param_fields()) plus any _extra_param_fields declared on the subclass.
Subclasses still define EpilogueArguments and epi_to_underlying_arguments
manually.
"""

from dataclasses import make_dataclass, MISSING

import cutlass.cute as cute
from cutlass import Int32, const_expr

from .cute_dsl_utils import ParamsBase
from .epi_ops import EpiContext, EpiSmemBytes, Scalar


def _make_epi_params(epi_ops, extra_fields, bases):
    # [FUNCTION DOC] HELPER: Create the EpilogueParams dataclass (named EpilogueParams, with the given base classes) from every op's
    # param_fields plus extra_fields, each a (name, type, default) triple. Fields without a default (MISSING) come first, then those
    # with defaults. Returns the new dataclass type.
    """Build EpilogueParams dataclass from epi_ops + extra fields.

    Required fields (default=MISSING) are placed first, then optional fields.
    """
    required, optional = [], []
    for op in epi_ops:
        for name, typ, default in op.param_fields():
            (required if default is MISSING else optional).append((name, typ, default))
    for name, typ, default in extra_fields:
        (required if default is MISSING else optional).append((name, typ, default))
    fields = [(n, t) for n, t, _ in required] + [(n, t, d) for n, t, d in optional]
    return make_dataclass("EpilogueParams", fields, bases=bases)


class ComposableEpiMixin:
    # [CLASS DOC] Mixin for GEMM classes that declares the epilogue as the class-level tuple _epi_ops (with optional
    # _extra_param_fields and _epi_param_bases for EpilogueParams). Provides host hooks (resolve_epi_m_major, epi_smem_bytes,
    # epi_get_smem_struct, epi_get_smem_tensors, epi_get_tma_atoms), device hooks (epi_begin, epi_begin_loop, epi_tile_load_*,
    # epi_end_loop, epi_end) that loop over ops. Per-instance _epi_ops is shadowed with only active (non-None arg) ops after
    # _epi_ops_to_params_dict runs; consumers should use .get(name) on the returned dicts.
    """Base mixin that composes EpiOps into the standard epilogue hooks."""

    _epi_ops = ()
    _extra_param_fields = ()  # [(name, type, default), ...] for non-op params (e.g. act_fn)
    _epi_param_bases = (ParamsBase,)  # Base classes for the auto-generated EpilogueParams

    def __init_subclass__(cls, **kwargs):
        # [FUNCTION DOC] HELPER: On subclass creation, if the subclass has non-empty _epi_ops and does not define EpilogueParams in its own
        # class dict, auto-generate and set EpilogueParams via _make_epi_params.
        super().__init_subclass__(**kwargs)
        if cls._epi_ops:
            # Auto-generate EpilogueParams if not explicitly defined on this class
            if "EpilogueParams" not in cls.__dict__:
                cls.EpilogueParams = _make_epi_params(
                    cls._epi_ops, cls._extra_param_fields, cls._epi_param_bases
                )

    # --- Host-side: args → params ---

    def _filter_epi_ops(self, args):
        # [FUNCTION DOC] HELPER: Set an instance attribute _epi_ops to the tuple of class-level ops whose same-named attribute on args is not
        # None. Mutates self.
        """Shadow `_epi_ops` with an instance-level tuple of only the ops whose
        arg is non-None. Called automatically by `_epi_ops_to_params_dict`, so
        subclass `epi_to_underlying_arguments` methods don't need to invoke it
        directly. After this runs, op hook methods can assume their
        `param`/`arg_tensor` is non-None."""
        self._epi_ops = tuple(
            op for op in type(self)._epi_ops if getattr(args, op.name, None) is not None
        )

    def _epi_ops_to_params_dict(self, args):
        # [FUNCTION DOC] Filter the ops to active ones (see _filter_epi_ops), then merge each active op's to_params result for args into one
        # dict, which is returned. Subclasses add custom fields and build EpilogueParams from it. Side effect: shadows _epi_ops on self.
        """Filter `_epi_ops` to active ops, then merge each op's to_params into
        a single dict. Subclasses call this from epi_to_underlying_arguments,
        add custom fields, then construct self.EpilogueParams(**d). Filtering
        here means every later iteration of self._epi_ops (host- and
        device-side) walks only active ops, and each op hook can assume its
        arg is non-None."""
        self._filter_epi_ops(args)
        d = {}
        for op in self._epi_ops:
            d.update(op.to_params(self, args))
        return d

    def resolve_epi_m_major(self, args):
        # [FUNCTION DOC] Decide whether the epilogue is M-major: sum each active op's epi_m_major_score (ops filtered inline from args since
        # this runs before epi_to_underlying_arguments) and return True when the total is non-negative.
        # Runs inside _setup_attributes, before epi_to_underlying_arguments,
        # because epi_m_major drives epi_tile / smem layout choices that
        # epi_to_underlying_arguments later consumes. self._epi_ops is still
        # the class-level schema at this point, so we filter inline from args
        # to keep op.epi_m_major_score's non-None invariant.
        score = 0
        for op in type(self)._epi_ops:
            arg = getattr(args, op.name, None)
            if arg is not None:
                score += op.epi_m_major_score(arg, self)
        return score >= 0

    # --- Host-side: smem allocation (queried from ops) ---

    @classmethod
    def epi_smem_bytes(cls, args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        # [FUNCTION DOC] Classmethod: total the shared-memory byte requirement of all active ops (args attribute non-None) for the given CTA
        # tile shape, epilogue tile and optional warp shape, accumulating into an EpiSmemBytes. Used to size pipeline stages.
        # Runs inside _compute_stages, before epi_to_underlying_arguments,
        # because the AB/epi stage counts (and therefore epi_c_stage) depend
        # on the epi smem budget that this returns; epi_to_underlying_arguments
        # then consumes epi_c_stage to build TileLoad's staged smem layout.
        # Stays a classmethod because _compute_stages is a classmethod and may
        # be invoked without an instance, so we filter inline from args.
        result = EpiSmemBytes()
        for op in cls._epi_ops:
            arg = getattr(args, op.name, None)
            if arg is not None:
                result += op.smem_bytes(arg, cta_tile_shape_mnk, epi_tile, warp_shape_mnk)
        return result

    def epi_get_smem_struct(self, params):
        # [FUNCTION DOC] Build the shared-storage struct type for the epilogue from each active op's smem_struct_field, sorted by size ascending
        # to limit alignment padding. Returns a zero-length Int32 placeholder range when no op contributes any smem.
        fields = []
        for op in self._epi_ops:
            result = op.smem_struct_field(self, params)
            if result is not None:
                fields.append(result)

        # cute.struct rejects empty annotations. When every active op contributes
        # no smem (e.g. only Scalar ops, or no active ops at all), return a
        # zero-byte placeholder matching gemm_base's default epi struct.
        if not fields:
            return cute.struct.MemRange[Int32, 0]

        # Sort smallest-to-largest so smaller fields pack ahead of larger
        # higher-aligned fields, reducing smem wasted to alignment padding.
        def _field_bytes(name_ftype):
            # [FUNCTION DOC] HELPER: Sort key giving the byte size of a cute struct containing the single (name, type) field.
            wrapper = type("_F", (), {"__annotations__": {name_ftype[0]: name_ftype[1]}})
            return cute.struct(wrapper).size_in_bytes()

        fields.sort(key=_field_bytes)
        annotations = {name: ftype for name, ftype in fields}
        EpiSharedStorage = type("EpiSharedStorage", (), {"__annotations__": annotations})
        return cute.struct(EpiSharedStorage)

    def epi_get_smem_tensors(self, params, storage):
        # [FUNCTION DOC] Return a dict mapping op name to the op's smem tensor taken from storage.epi, for all active ops except Scalar ops.
        return {
            op.name: op.get_smem_tensor(self, params, storage.epi)
            for op in self._epi_ops
            if not isinstance(op, Scalar)
        }

    def epi_get_tma_atoms(self, params, *, loc=None, ip=None):
        # [FUNCTION DOC] Return a flat list of the TMA atoms contributed by all active ops (loc and ip are unused).
        atoms = []
        for op in self._epi_ops:
            atoms.extend(op.tma_atoms(self, params))
        return atoms

    # --- Device-side: kernel execution (delegates to ops) ---

    @cute.jit
    def epi_begin(
        self,
        params,
        epi_smem_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        epilogue_barrier,
        tidx,
        tRS_rD_layout=None,
    ):
        # [FUNCTION DOC] Device-side epilogue start: build an EpiContext from the tiling, copy, barrier and thread arguments, call each active
        # op's begin, and return a dict of results keyed by op name. If any op needs an async fence, commits and waits on cp.async groups
        # and syncs on the epilogue barrier before returning.
        ctx = EpiContext(
            self,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            epilogue_barrier,
            tidx,
            tRS_rD_layout,
        )
        results = {
            op.name: op.begin(
                self,
                getattr(params, op.name),
                epi_smem_tensors.get(op.name),
                ctx,
            )
            for op in self._epi_ops
        }
        # self._epi_ops is filtered to active ops, so any op needing a fence
        # has a non-None tensor; no inner None check required.
        has_async_data = any(op.needs_async_fence() for op in self._epi_ops)
        if const_expr(has_async_data):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            epilogue_barrier.arrive_and_wait()
        return results

    def epi_begin_loop(self, params, epi_tensors, epi_coord):
        # [FUNCTION DOC] Per-subtile start: return a dict mapping op name to that op's begin_loop result for its tensors and the epi_coord.
        return {
            op.name: op.begin_loop(self, epi_tensors[op.name], epi_coord) for op in self._epi_ops
        }

    def epi_tile_load_g2s_copy_fns(
        self,
        params,
        epi_smem_tensors,
        tile_coord_mnkl,
        varlen_manager,
        epi_pipeline,
    ):
        # [FUNCTION DOC] Return a tuple of global-to-shared copy functions, one per active tile-load op (ops where is_tile_load is true).
        return tuple(
            op.load_g2s_copy_fn(
                self,
                params,
                epi_smem_tensors.get(op.name),
                tile_coord_mnkl,
                varlen_manager,
                epi_pipeline,
            )
            for op in self._epi_ops
            if op.is_tile_load()
        )

    @cute.jit
    def epi_tile_load_s2r(self, params, epi_tensors, stage_idx):
        # [FUNCTION DOC] Device-side: for each active op, load its shared-memory data into registers for the given pipeline stage index.
        for op in self._epi_ops:
            op.load_s2r(self, getattr(params, op.name), epi_tensors[op.name], stage_idx)

    @cute.jit
    def epi_end_loop(
        self,
        params,
        epi_tensors,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        # [FUNCTION DOC] Device-side: call each active op's end_loop with its params and tensors after processing a subtile. Returns nothing.
        for op in self._epi_ops:
            op.end_loop(
                self,
                getattr(params, op.name),
                epi_tensors[op.name],
                epi_coord,
                epi_tile,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tile_coord_mnkl,
                varlen_manager,
                tidx,
            )

    @cute.jit
    def epi_end(
        self,
        params,
        epi_tensors,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        # [FUNCTION DOC] Device-side: call each active op's end hook once all subtiles of the tile are done. Returns nothing.
        for op in self._epi_ops:
            op.end(
                self,
                getattr(params, op.name),
                epi_tensors[op.name],
                epi_tile,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tile_coord_mnkl,
                varlen_manager,
                tidx,
            )
