# [MODULE DOC] HELPER: Standalone script used by compiler bisector tests. It compiles a small
# function using in-place exponential_ on a GPU tensor while the inductor decomposition table is
# patched so that exponential produces NaN, and exits with code 1 if the output has NaN, else 0.
# Importing has no side effects beyond defining the aten alias and functions.
import os
from contextlib import contextmanager

import torch
import torch._prims_common as utils
from torch.testing._internal.inductor_utils import GPU_TYPE


aten = torch.ops.aten


def bad_exp_decomp(self, rate=1.0, generator=None):
    # [FUNCTION DOC] HELPER: Deliberately broken decomposition of exponential. Raises
    # AssertionError if a generator is passed, checks (via torch._check) that the dtype is a real
    # floating type and that rate is positive, then returns random values multiplied by NaN, so
    # the result is a tensor of NaN shaped like the input.
    if generator is not None:
        raise AssertionError("Expected generator to be None")
    torch._check(
        not utils.is_complex_dtype(self.dtype) and utils.is_float_dtype(self.dtype),
        lambda: (
            "Exponential distribution is a continuous probability distribution. "
            f"dtype must be a floating point but you specified {self.dtype}."
        ),
    )
    torch._check(
        rate > 0.0,
        lambda: f"exponential_ expects lambda > 0.0, but found lambda={rate}",
    )
    return torch.rand_like(self) * float("nan")


@contextmanager
def patch_exp_decomp():
    # [FUNCTION DOC] HELPER: Context manager that replaces the select_decomp_table attribute of
    # torch._inductor.compile_fx with a function returning a copy of the normal table with the
    # aten exponential default overload mapped to the bad decomposition. The original function is
    # restored on exit, also when the body raises.
    from torch._inductor.compile_fx import select_decomp_table as old_decomp

    def get_decomp():
        # [FUNCTION DOC] HELPER: Return a copy of the original decomposition table with the
        # exponential entry overridden by the NaN-producing decomposition.
        out = old_decomp().copy()
        out[aten.exponential.default] = bad_exp_decomp
        return out

    try:
        torch._inductor.compile_fx.select_decomp_table = get_decomp
        yield
    finally:
        torch._inductor.compile_fx.select_decomp_table = old_decomp


def vq(x):
    # [FUNCTION DOC] HELPER: Function under test; adds 3, applies in-place exponential_, then
    # scales by 10.5.
    return (x + 3).exponential_() * 10.5


def test_fn():
    # [FUNCTION DOC] HELPER: With the bad decomposition patched, compile vq and run it on a random
    # (4, 400, 256) tensor on the GPU device type. Returns 1 if the output contains any NaN
    # (the decomposition was used), else 0. Used as the process exit status.
    with patch_exp_decomp():
        vq_compiled = torch.compile(vq)  # noqa: UNSPECIFIED_BACKEND
        x = torch.randn(4, 400, 256, device=GPU_TYPE)
        out_compiled = vq_compiled(x)

    return 1 if out_compiled.isnan().any() else 0


if __name__ == "__main__":
    os._exit(test_fn())
