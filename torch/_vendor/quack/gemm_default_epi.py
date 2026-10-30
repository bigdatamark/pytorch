# Copyright (c) 2025, Wentao Guo, Tri Dao.
# [MODULE DOC] Default GEMM epilogue for quack: scaling by alpha, optional addition of C
# scaled by beta, and optional row/column vector broadcast addition. Provides the mixin
# GemmDefaultEpiMixin and concrete kernel classes for SM80, SM90, SM100 and SM120 formed by
# combining it with the corresponding GEMM class. No import-time side effects.
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from .cute_dsl_utils import mlir_namedtuple
from .epi_composable import ComposableEpiMixin
from .epi_ops import Scalar, RowVecLoad, ColVecLoad
from .gemm_sm80 import GemmSm80
from .gemm_sm90 import GemmSm90
from .gemm_sm100 import GemmSm100
from .gemm_sm120 import GemmSm120
from .rounding import RoundingMode
from . import layout_utils
from . import utils


class GemmDefaultEpiMixin(ComposableEpiMixin):
    # [CLASS DOC] Composable epilogue mixin implementing D = alpha * acc + beta * C plus
    # broadcast row and column vectors. Declares its epilogue operands (scalars alpha, beta
    # and the stochastic rounding seed, a row vector load and a column vector load) and an
    # argument tuple type. The parameter type is generated from those operands. Stores the
    # rounding mode on the instance when arguments are converted.
    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        RowVecLoad("mRowVecBroadcast"),
        ColVecLoad("mColVecBroadcast"),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # [CLASS DOC] Named tuple of user epilogue arguments, all optional: alpha and beta
        # (scalar or tensor), row and column broadcast vectors, compile-time add_to_output
        # flag (default false), compile-time rounding mode (default round to nearest) and
        # sr_seed for stochastic rounding.
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        # [FUNCTION DOC] Converts epilogue arguments into kernel parameters: saves the
        # rounding mode on self, builds the parameter dictionary from the declared operands,
        # and for the row and column vector entries that are present and listed in the
        # instance concat layout rewrites the tensor from concatenated to interleaved layout
        # along mode 1. Returns the parameter object built from the dictionary.
        self.rounding_mode = args.rounding_mode
        d = self._epi_ops_to_params_dict(args)
        for key in ("mRowVecBroadcast", "mColVecBroadcast"):
            if key in self.concat_layout and key in d:
                d[key] = layout_utils.concat_to_interleave(d[key], 1)
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        # [FUNCTION DOC] Per sub-tile compute (JIT). Loads the accumulator fragment,
        # multiplies by alpha if params has a non-None alpha (read as a scalar or through a
        # pointer), then if a C fragment is present adds it converted to the D element type,
        # scaled by beta when beta is set, else with unit weight. Stores the result back into
        # the D fragment, then adds the row vector and column vector fragments element by
        # element when they are active. Returns None (no aux output). Mutates the D register
        # fragment in place.
        # Use .get(): inactive ops are filtered out of epi_loop_tensors.
        alpha = epi_loop_tensors.get("alpha")
        beta = epi_loop_tensors.get("beta")
        tDrRowVec = epi_loop_tensors.get("mRowVecBroadcast")
        tDrColVec = epi_loop_tensors.get("mColVecBroadcast")
        rD = tRS_rD.load()
        # Apply alpha scaling to accumulator if alpha is provided (not None)
        if const_expr(hasattr(params, "alpha") and params.alpha is not None):
            alpha = utils.load_scalar_or_pointer(params.alpha)
            rD *= alpha
        # Apply C with beta scaling
        if const_expr(tRS_rC is not None):
            if const_expr(not hasattr(params, "beta") or params.beta is None):
                # beta is None, default behavior: add C (beta=1.0)
                rD += tRS_rC.load().to(tRS_rD.element_type)
            else:
                beta = utils.load_scalar_or_pointer(params.beta)
                rD += beta * tRS_rC.load().to(tRS_rD.element_type)
        tRS_rD.store(rD)
        if const_expr(tDrRowVec is not None):
            for i in cutlass.range(cute.size(tDrRowVec), unroll_full=True):
                tRS_rD[i] += tDrRowVec[i]
        if const_expr(tDrColVec is not None):
            for i in cutlass.range(cute.size(tDrColVec), unroll_full=True):
                tRS_rD[i] += tDrColVec[i]
        return None


class GemmDefaultSm80(GemmDefaultEpiMixin, GemmSm80):
    # [CLASS DOC] SM80 GEMM kernel with the default epilogue mixin.
    pass


class GemmDefaultSm90(GemmDefaultEpiMixin, GemmSm90):
    # [CLASS DOC] SM90 GEMM kernel with the default epilogue mixin.
    pass


class GemmDefaultSm100(GemmDefaultEpiMixin, GemmSm100):
    # [CLASS DOC] SM100 GEMM kernel with the default epilogue mixin.
    pass


class GemmDefaultSm120(GemmDefaultEpiMixin, GemmSm120):
    # [CLASS DOC] SM120 GEMM kernel with the default epilogue mixin.
    pass
