# mypy: disable-error-code=import-not-found
# [MODULE DOC] CuteDSL jit helper functions used by generated kernels to convert
# between SSA tensor values and non-SSA register fragments/scalars (needed for
# indexed loads). Requires the cutlass package at import time.
# pyrefly: ignore [import-error, missing-import]
import cutlass
import cutlass.cute as cute


@cute.jit  # type: ignore[misc]
def ssa_to_indexable(ssa_value: cute.TensorSSA, dtype: str) -> cute.Numeric:
    # [FUNCTION DOC] HELPER: Converts an SSA vector to a scalar usable as an index
    # by materializing it into a register fragment (cast to dtype) and returning
    # its first element. Workaround for missing gather support.
    """
    Convert SSA form to indexable non-SSA form.

    Workaround for lack of gather support: SSA values cannot be used directly
    as indices in tensor loads. This converts SSA to a register fragment, then
    extracts a scalar for indexing.
    """
    frag = ssa_to_fragment(ssa_value, dtype)
    return frag[0]


@cute.jit  # type: ignore[misc]
def ssa_to_fragment(ssa_value: cute.TensorSSA, dtype: str) -> cute.Tensor:
    # [FUNCTION DOC] HELPER: Stores an SSA vector into a new register-memory tensor
    # of the same shape and given dtype, casting first (at compile time) if the SSA
    # dtype differs. Returns the fragment.
    """Materialize an SSA vector into a register fragment.

    Casts to the fragment dtype when the SSA dtype differs (e.g. an int64 index
    expression materialized into an int32 index fragment), since fragment
    stores require matching element types.
    """
    if cutlass.const_expr(ssa_value.dtype != dtype):
        ssa_value = ssa_value.to(dtype)
    frag = cute.make_rmem_tensor(ssa_value.shape, dtype)
    frag.store(ssa_value)
    return frag


@cute.jit  # type: ignore[misc]
def result_to_ssa(value: cute.Numeric, dtype: str) -> cute.TensorSSA:
    # [FUNCTION DOC] HELPER: Wraps a scalar value in a one-element register
    # fragment of the given dtype and loads it back as an SSA value.
    """
    Convert non-SSA result back to SSA form.

    After performing operations with non-SSA values (like indexed loads),
    convert the result back to SSA form for further computation.
    """
    frag = cute.make_rmem_tensor(1, dtype)
    frag[0] = value
    return frag.load()
