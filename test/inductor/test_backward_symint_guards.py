# Owner(s): ["module: inductor"]
# [MODULE DOC] Regression test for FxGraphCache guard lookup of a backward graph that only
# sees a dynamic batch size through a compound symbolic expression (batch plus one).
# Defines two custom ops (a dense-to-jagged style forward and a jagged-to-padded-dense
# style backward), registers their fake kernels and autograd formula at import time under
# the test_bro namespace, and one test case.
import torch
import torch._functorch.config as functorch_config
import torch._inductor.config as inductor_config
from torch import Tensor
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_caches, fresh_cache


# A minimal opaque op pair modeling fbgemm.dense_to_jagged (fwd) /
# jagged_to_padded_dense (bwd). The backward reconstructs the dense shape
# *inside* the op from the (B_RO + 1)-sized offsets tensor, so the batch symbol
# B_RO only ever reaches the backward graph inside the compound expression
# `B_RO + 1` -- it is never a standalone SymInt input. This is the structure
# that triggers the bug; see test below.
@torch.library.custom_op("test_bro::d2j", mutates_args=())
def _d2j(dense: Tensor, offsets: Tensor, nnz: int) -> Tensor:
    # [FUNCTION DOC] Custom op test_bro::d2j (no mutation). Flattens the leading two dims of
    # the dense input and returns the first nnz rows as a contiguous 2D tensor. The offsets
    # argument is accepted but unused. Raises if dense is not 3D (unpacking failure).
    b, length, d = dense.shape
    return dense.reshape(b * length, d)[:nnz].contiguous()


@_d2j.register_fake
def _(dense, offsets, nnz):
    # [FUNCTION DOC] HELPER: Fake (meta) kernel for d2j; returns an uninitialized tensor of
    # shape nnz by the last dim of dense, without reading offsets.
    return dense.new_empty((nnz, dense.shape[2]))


@torch.library.custom_op("test_bro::j2d", mutates_args=())
def _j2d(grad: Tensor, offsets: Tensor, length: int, nnz: int) -> Tensor:
    # [FUNCTION DOC] Custom op test_bro::j2d (no mutation). Derives the batch size from the
    # offsets length minus one, builds a zero tensor with batch times length rows, copies the first
    # nnz rows from grad, and returns it reshaped to batch, length, feature dims. The batch
    # size therefore only reaches the compiled backward via the offsets size.
    b = offsets.shape[0] - 1
    out = grad.new_zeros(b * length, grad.shape[1])
    out[:nnz] = grad
    return out.reshape(b, length, grad.shape[1])


@_j2d.register_fake
def _(grad, offsets, length, nnz):
    # [FUNCTION DOC] HELPER: Fake (meta) kernel for j2d; returns an uninitialized tensor of
    # shape batch (offsets length minus one), length, grad feature dim.
    b = offsets.shape[0] - 1
    return grad.new_empty((b, length, grad.shape[1]))


def _d2j_setup(ctx, inputs, output):
    # [FUNCTION DOC] HELPER: Autograd setup_context for d2j. Saves offsets for backward and
    # stores the dense input's second dimension (as length) and nnz on the context.
    dense, offsets, nnz = inputs
    ctx.save_for_backward(offsets)
    ctx.length = dense.shape[1]
    ctx.nnz = nnz


def _d2j_backward(ctx, grad):
    # [FUNCTION DOC] HELPER: Autograd backward for d2j. Retrieves saved offsets and returns
    # the j2d result for grad, plus None gradients for offsets and nnz.
    (offsets,) = ctx.saved_tensors
    return torch.ops.test_bro.j2d(grad, offsets, ctx.length, ctx.nnz), None, None


torch.library.register_autograd(
    "test_bro::d2j", _d2j_backward, setup_context=_d2j_setup
)


class TestBackwardSymIntGuardBinding(TestCase):
    # [CLASS DOC] Test case for the cache-lookup bug where a compound symbolic batch size in the
    # backward graph guards could not be bound to arguments by FxGraphCache.
    @inductor_config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "compile_threads": 1,
        }
    )
    @functorch_config.patch({"enable_autograd_cache": False})
    def test_backward_compound_symint_guard_is_bindable(self):
        # [FUNCTION DOC] Regression test. Under a fresh inductor cache, compiles a small model that
        # expands a parameter to a dynamic batch size and calls d2j, then runs forward and backward
        # with batch 7 and then 5 (second run triggers automatic dynamic shapes and stores the
        # dynamic backward). Clears in-memory caches and resets dynamo, reruns with 7 (which must
        # not raise a KeyError during guarded-entry lookup) and checks the gradient for 5 equals
        # the eager gradient. Patches fx_graph_cache on, remote cache off, one compile thread, and
        # disables the autograd cache.
        # Regression test for the partitioner passing a raw (compound) SymInt
        # binding to the backward: a dynamic scalar int `B_RO` (Dynamo source
        # L['B_RO']) is used as pos.expand(B_RO, -1, -1) and its only backward
        # relevance is the saved offsets tensor of size B_RO + 1. The backward
        # then references B_RO only through `B_RO + 1`, so FxGraphCache's stored
        # guards_expr renders it via the var_to_sources fallback as L['B_RO'].
        # On a later cache lookup, find_guarded_entry ->
        # evaluate_guards_expression evaluates that expression with args named
        # t0, t1, ... and raises KeyError: 'B_RO'.
        length, dim = 64, 8

        def model(pos, b_ro, offsets, nnz):
            # [FUNCTION DOC] HELPER: Model under test; views pos as 1 by length by dim, expands the
            # batch dim to b_ro, applies d2j and returns the sum.
            x = pos.view(1, length, dim).expand(b_ro, -1, -1)
            return torch.ops.test_bro.d2j(x, offsets, nnz).sum()

        def make(b_ro):
            # [FUNCTION DOC] HELPER: Builds inputs for a batch size: a random leaf parameter,
            # the batch size, int64 offsets (arange of batch plus one, times three) and nnz equal
            # to three times batch.
            pos = torch.randn(length * dim, requires_grad=True)
            offsets = torch.arange(b_ro + 1, dtype=torch.int64) * 3
            return pos, b_ro, offsets, b_ro * 3

        def run(fn, b_ro):
            # [FUNCTION DOC] HELPER: Runs fn on inputs for a batch size, backpropagates the summed
            # result and returns the gradient of the parameter.
            pos, b, offsets, nnz = make(b_ro)
            fn(pos, b, offsets, nnz).sum().backward()
            return pos.grad

        with fresh_cache():
            compiled = torch.compile(model)
            # First value specializes B_RO; the second promotes it to dynamic
            # (automatic dynamic) and stores the dynamic backward entry.
            run(compiled, 7)
            run(compiled, 5)

            # Drop in-memory caches so the next compile re-looks-up the stored
            # dynamic backward via find_guarded_entry (mirrors a second rank
            # reading the shared cache).
            clear_caches()
            torch._dynamo.reset()

            # This lookup must not raise KeyError, and must match eager grads.
            run(compiled, 7)
            self.assertEqual(run(compiled, 5), run(model, 5))


if __name__ == "__main__":
    run_tests()
