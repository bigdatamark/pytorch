# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that Helion kernels (the helion DSL) work on their own and when called from
# a torch.compile region using the inductor backend. Helion is imported only when HAS_HELION is
# true. Tests need triton and helion and run on the GPU_TYPE device. No import-time side effects
# beyond the optional helion import.
import torch
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_HELION,
    requires_helion,
    requires_triton,
)


if HAS_HELION:
    import helion
    import helion.language as hl


class HelionTests(TestCase):
    # [CLASS DOC] Test case holding two Helion kernel tests: an elementwise add kernel run
    # eagerly and under fullgraph inductor compilation, and a row softmax kernel checked against
    # torch softmax.
    @requires_triton()
    @requires_helion()
    def test_add_kernel(self):
        # [FUNCTION DOC] Defines a Helion add kernel with a fixed config of block sizes one and
        # two that broadcasts both inputs, promotes dtype, and writes tile-wise sums. Runs it on
        # random half precision GPU tensors both directly and through a fullgraph inductor
        # compiled wrapper, and asserts both results equal the plain tensor sum.
        @helion.kernel(config=helion.Config(block_sizes=[1, 2]))
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Helion kernel adding two tensors with broadcasting and type
            # promotion matching torch add; loops over tiles of the output and stores the tile
            # sums. Returns the newly allocated output tensor.
            # match pytorch broadcasting rules
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty(
                x.shape,
                # match type promotion of torch.add
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            # tile will be a tuple of blocks
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Thin wrapper that calls the Helion add kernel, used as the
            # function that gets compiled with torch.compile.
            return add(x, y)

        x = torch.randn(4, 8, device=GPU_TYPE, dtype=torch.float16)
        y = torch.randn(4, 8, device=GPU_TYPE, dtype=torch.float16)

        out = add(x, y)
        compiled_add = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_add(x, y)

        self.assertEqual(out, x + y)
        self.assertEqual(compiled_out, x + y)

    @requires_triton()
    @requires_helion()
    def test_softmax_view_reshape(self):
        # [FUNCTION DOC] Runs a Helion softmax kernel (block size one) on a 1024 by 1024 half
        # precision GPU tensor. The kernel exercises view and reshape on reduced values inside a
        # tile loop. Asserts the result matches torch.nn.functional.softmax over dim one within
        # loose tolerances.
        @helion.kernel(config={"block_size": 1})
        def softmax(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Helion kernel computing a row-wise softmax. For each tile
            # of rows it subtracts the row max (reshaped via view), exponentiates, divides by
            # the row sum (reshaped) and stores into an output allocated like the input. Returns
            # that output.
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1).view(tile_n, 1)
                exp = torch.exp(values - amax)
                sum_exp = torch.reshape(torch.sum(exp, dim=1), [tile_n, 1])
                out[tile_n, :] = exp / sum_exp
            return out

        x = torch.randn([1024, 1024], device=GPU_TYPE, dtype=torch.float16)
        result = softmax(x)
        self.assertEqual(
            result, torch.nn.functional.softmax(x, dim=1), rtol=1e-2, atol=1e-1
        )


instantiate_parametrized_tests(HelionTests)


if __name__ == "__main__":
    run_tests()
