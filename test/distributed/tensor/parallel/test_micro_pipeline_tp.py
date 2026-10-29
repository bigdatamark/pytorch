# Owner(s): ["module: c10d"]
# [MODULE DOC] Tests for Inductor's micro-pipeline tensor-parallel pass, which fuses
# all-gather + matmul and matmul + reduce-scatter (plain and fp8 scaled) into symmetric-memory
# fused ops. Uses a fake process group (no real communication) with a CUDA device, inspects
# FX graphs and generated Triton code, and includes a DTensor sequence-parallel end-to-end case.
# Defines the helpers _make_post_grad_fx and _fp8_all_gather and the test classes
# MicroPipelineTPTest (world size 2) and MicroPipelineTP4GPUTest (world size 4).
import unittest

import torch
import torch.distributed as dist
from functorch import make_fx
from torch._inductor.decomposition import decompositions
from torch._inductor.fx_passes.micro_pipeline_tp import (
    _get_unexposed_collectives,
    find_all_gather_patterns,
    find_reduce_scatter_patterns,
    micro_pipeline_tp_pass,
)
from torch._inductor.fx_passes.post_grad import remove_noop_ops, view_to_reshape
from torch._inductor.utils import fresh_cache, run_and_get_triton_code
from torch.distributed._functional_collectives import (
    all_gather_single,
    all_gather_tensor,
    reduce_scatter_single,
    reduce_scatter_tensor,
)
from torch.distributed._symmetric_memory import _test_mode
from torch.distributed.distributed_c10d import _get_group_size_by_name
from torch.distributed.tensor import DeviceMesh, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8
from torch.testing._internal.common_device_type import e4m3_type
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    run_tests,
    TEST_WITH_ROCM,
    TEST_WITH_TORCHINDUCTOR,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.inductor_utils import HAS_GPU


def _make_post_grad_fx(f, *inps):
    # [FUNCTION DOC] HELPER: Traces f with fake tensors into an FX graph module using the
    # Inductor decompositions, then removes no-op ops and rewrites views to reshapes (in place
    # on the graph) so it resembles a post-grad Inductor graph. Returns the graph module.
    gm = make_fx(f, decompositions, tracing_mode="fake")(*inps)
    remove_noop_ops(gm.graph)
    view_to_reshape(gm)
    return gm


def _fp8_all_gather(
    tensor: torch.Tensor, gather_dim: int, group_name: str
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: All-gathers an fp8 tensor by gathering on dim 0 (as a function
    # of group name) and viewing back to the input dtype. For gather_dim other than 0 it
    # splits the result into one chunk per rank, views each chunk as uint8 and concatenates
    # along gather_dim before viewing back to the original dtype. Returns the gathered tensor.
    # We don't yet have a canonical pattern for fp8 all-gather. This is a
    # pattern observed in DTensor + float8_experimental.
    ag = all_gather_single(tensor, gather_dim=0, group=group_name)
    if gather_dim == 0:
        return ag.view(tensor.dtype)
    chunks = ag.chunk(_get_group_size_by_name(group_name))
    chunks = [chunk.view(torch.uint8) for chunk in chunks]
    return torch.cat(chunks, dim=gather_dim).view(tensor.dtype)


@instantiate_parametrized_tests
class MicroPipelineTPTest(TestCase):
    # [CLASS DOC] Tests the micro-pipeline TP pass with a fake 2-rank process group (rank 0).
    # setUp enables the micro-pipeline Inductor config flag and initializes the fake group;
    # tearDown destroys it. Tests are skipped when no GPU/Triton is available.
    def setUp(self):
        # [FUNCTION DOC] Sets the global Inductor micro-pipeline flag to enabled (not restored
        # later), fixes rank 0 and world size 2, selects cuda:0 and initializes a default
        # process group with the fake backend and a FakeStore.
        super().setUp()
        torch._inductor.config._micro_pipeline_tp = True

        self.rank = 0
        self.world_size = 2
        torch.cuda.set_device("cuda:0")

        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    def tearDown(self):
        # [FUNCTION DOC] Destroys the default process group created in setUp.
        # KNOWN ISSUE: does not call the parent tearDown, and the Inductor flag set in setUp
        # is never reset.
        dist.destroy_process_group()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_find_all_gather_patterns(self):
        # [FUNCTION DOC] Traces a function with six all-gather variants (dim 0, dim 1, two fp8
        # variants, and two cat-of-narrow variants, one followed by a slice) and checks
        # find_all_gather_patterns returns six matches, each rooted at the all_gather_into_tensor
        # op in the world group, with the expected gather dim and result node kind (wait,
        # cat, view dtype or slice).
        group = dist.group.WORLD

        def func(
            inp: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            # [FUNCTION DOC] HELPER: Returns six differently shaped all-gathers of inp.
            a = all_gather_single(inp, gather_dim=0, group=group.group_name)
            b = all_gather_single(inp, gather_dim=1, group=group.group_name)
            c = _fp8_all_gather(inp, gather_dim=0, group_name=group.group_name)
            d = _fp8_all_gather(inp, gather_dim=1, group_name=group.group_name)
            e_full = all_gather_tensor(inp, gather_dim=0, group=group.group_name)
            e = torch.cat(
                [e_full.narrow(0, 0, 64), e_full.narrow(0, 64, 64)],
                dim=1,
            )
            f_full = all_gather_tensor(inp, gather_dim=0, group=group.group_name)
            f_cat = torch.cat(
                [f_full.narrow(0, 0, 64), f_full.narrow(0, 64, 64)],
                dim=1,
            )
            f = f_cat.narrow(1, 0, 32)
            return a, b, c, d, e, f

        inp = torch.rand(64, 32, device="cuda")

        gm = _make_post_grad_fx(func, inp)
        all_gathers = find_all_gather_patterns(gm.graph)
        self.assertEqual(len(all_gathers), 6)

        # If this test fails, please update find_all_gather_patterns instead of
        # modifying the following assertions.
        for all_gather in all_gathers:
            self.assertEqual(
                all_gather.ag_node.target,
                torch.ops._c10d_functional.all_gather_into_tensor.default,
            )
            self.assertEqual(all_gather.group_name, group.group_name)

        self.assertEqual(all_gathers[0].gather_dim, 0)
        self.assertEqual(
            all_gathers[0].res_node.target,
            torch.ops._c10d_functional.wait_tensor.default,
        )

        self.assertEqual(all_gathers[1].gather_dim, 1)
        self.assertEqual(
            all_gathers[1].res_node.target,
            torch.ops.aten.cat.default,
        )

        self.assertEqual(all_gathers[2].gather_dim, 0)
        self.assertEqual(
            all_gathers[2].res_node.target,
            torch.ops._c10d_functional.wait_tensor.default,
        )

        self.assertEqual(all_gathers[3].gather_dim, 1)
        self.assertEqual(
            all_gathers[3].res_node.target,
            torch.ops.aten.view.dtype,
        )

        self.assertEqual(all_gathers[4].gather_dim, 1)
        self.assertEqual(
            all_gathers[4].res_node.target,
            torch.ops.aten.cat.default,
        )

        self.assertEqual(all_gathers[5].gather_dim, 1)
        self.assertEqual(
            all_gathers[5].res_node.target,
            torch.ops.aten.slice.Tensor,
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_find_reduce_scatter_patterns(self):
        # [FUNCTION DOC] Traces three reduce-scatters (sum on dim 0, avg on dim 1, and avg after
        # a cat of narrowed slices) and checks find_reduce_scatter_patterns finds three matches
        # whose input is a placeholder, with the reduce_scatter_tensor and wait_tensor nodes,
        # world group name, and the expected reduce op and scatter dim per match.
        group = dist.group.WORLD

        def func(
            inp: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Returns three differently shaped reduce-scatters of inp.
            a = reduce_scatter_single(inp, "sum", scatter_dim=0, group=group.group_name)
            b = reduce_scatter_single(inp, "avg", scatter_dim=1, group=group.group_name)
            c_inp = torch.cat(
                [inp.narrow(1, 0, 16), inp.narrow(1, 16, 16)],
                dim=0,
            )
            c = reduce_scatter_tensor(
                c_inp, "avg", scatter_dim=0, group=group.group_name
            )
            return a, b, c

        inp = torch.rand(64, 32, device="cuda")

        gm = make_fx(func)(inp)
        reduce_scatters = find_reduce_scatter_patterns(gm.graph)
        self.assertEqual(len(reduce_scatters), 3)

        # If this test fails, please update find_reduce_scatter_patterns
        # instead of modifying the following assertions.
        for reduce_scatter in reduce_scatters:
            self.assertEqual(
                reduce_scatter.input_node.op,
                "placeholder",
            )
            self.assertEqual(
                reduce_scatter.reduce_scatter_node.target,
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
            )
            self.assertEqual(
                reduce_scatter.wait_tensor_node.target,
                torch.ops._c10d_functional.wait_tensor.default,
            )
            self.assertEqual(reduce_scatter.group_name, group.group_name)

        self.assertEqual(reduce_scatters[0].reduce_op, "sum")
        self.assertEqual(reduce_scatters[0].scatter_dim, 0)
        self.assertEqual(reduce_scatters[2].reduce_op, "avg")
        self.assertEqual(reduce_scatters[2].scatter_dim, 1)

        self.assertEqual(reduce_scatters[1].reduce_op, "avg")
        self.assertEqual(reduce_scatters[1].scatter_dim, 1)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_get_unexposed_collectives(self):
        # [FUNCTION DOC] Traces a graph where an all-gather and a reduce-scatter each overlap
        # with a matmul while a final all-gather has nothing to hide behind, and checks
        # _get_unexposed_collectives returns just the first two (all_gather_into_tensor then
        # reduce_scatter_tensor).
        group = dist.group.WORLD

        def func(inp: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns three outputs mixing matmuls and collectives.
            # KNOWN ISSUE: the annotation says a single tensor but a tuple is returned.
            a = inp @ inp.T
            # b is unexposed (hidden by a)
            b = all_gather_single(inp, gather_dim=0, group=group.group_name)
            c = b @ inp.T
            # d is unexposed (hidden by c)
            d = reduce_scatter_single(b, "avg", scatter_dim=0, group=group.group_name)
            # e is exposed
            e = all_gather_single(d, gather_dim=0, group=group.group_name)
            return a, c, e

        inp = torch.rand(64, 32, device="cuda")

        gm = make_fx(func)(inp)
        overlappable_collectives = _get_unexposed_collectives(gm.graph)
        self.assertEqual(
            list(map(str, overlappable_collectives)),
            ["all_gather_into_tensor", "reduce_scatter_tensor"],
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @parametrize("A_dims", [2, 3])
    @parametrize("gather_dim", [0, 1, 2])
    @parametrize("return_A", [True, False])
    @fresh_cache()
    def test_fuse_all_gather_matmul(self, A_dims, gather_dim, return_A):
        # [FUNCTION DOC] Parametrized over input rank, gather dim and whether the gathered
        # input is also returned; combinations with gather_dim out of range are skipped.
        # Compiles all-gather + matmul under symmetric-memory test mode, checks eager and
        # compiled output strides match, and checks the generated code contains the fused op
        # (and no plain all-gather, and return_A set accordingly) unless gathering on the matmul
        # contraction dim, where fusion must not occur.
        if gather_dim >= A_dims:
            return

        group = dist.group.WORLD

        def func(A_shard: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: All-gathers A_shard on gather_dim and multiplies by B;
            # returns a pair of the gathered A (or None) and the product.
            A = all_gather_single(A_shard, gather_dim=gather_dim, group=group)
            if return_A:
                return A, A @ B
            else:
                return None, A @ B

        if A_dims == 2:
            A_shard_shape = [64, 32]
        elif A_dims == 3:
            A_shard_shape = [2, 64, 32]
        else:
            raise AssertionError(f"Invalid A_dims: {A_dims}")

        A_shard_shape[gather_dim] //= self.world_size
        A_shard = torch.rand(*A_shard_shape, device="cuda")
        B = torch.rand(32, 16, device="cuda")

        with _test_mode():
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, A_shard, B)

            eager_stride = func(A_shard, B)[1].stride()
            compiled_stride = compiled(A_shard, B)[1].stride()
            self.assertEqual(eager_stride, compiled_stride)

        if gather_dim == A_dims - 1:
            # Gathering on the K dimension of the matmul -- fusion not supported.
            self.assertNotIn("fused_all_gather_matmul", code)
        else:
            self.assertIn("fused_all_gather_matmul", code)
            self.assertNotIn("all_gather_into_tensor", code)
            self.assertEqual("return_A=True" in code, return_A)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_fuse_all_gather_matmul_view_optimization(self):
        # [FUNCTION DOC] With a leading batch dim of 1 and gather on dim 1, the gather becomes a
        # view so no all-gather/matmul pattern remains; compiles under symmetric-memory test
        # mode and asserts the fused op does not appear in the generated code.
        """When batch=1 and gather_dim=1, _maybe_view_chunk_cat uses a view
        (no data movement), so the all_gather is optimized away entirely and
        there is no all_gather+matmul pattern to fuse."""
        group = dist.group.WORLD

        def func(A_shard: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: All-gathers on dim 1 and multiplies by B; returns product.
            A = all_gather_single(A_shard, gather_dim=1, group=group)
            return A @ B

        # batch=1: after all_gather, shape[0] == world_size == group_size,
        # so the view optimization in _maybe_view_chunk_cat applies.
        # Shard is [1, 32, 32], all_gather gives [2, 32, 32], view to [1, 64, 32].
        A_shard = torch.rand(1, 32, 32, device="cuda")
        B = torch.rand(32, 16, device="cuda")

        with _test_mode():
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, A_shard, B)

        self.assertNotIn("fused_all_gather_matmul", code)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_fuse_all_gather_matmul_slice_cat(self):
        # [FUNCTION DOC] Builds a post-grad graph where a dim-0 all-gather is rearranged to dim 1
        # by narrowing and concatenating, with an unbacked leading dimension, runs the
        # micro-pipeline pass under symmetric-memory test mode and asserts the graph now has the
        # fused all-gather-matmul op and no plain all-gather.
        group = dist.group.WORLD

        def func(A_shard: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Gathers on dim 0, re-lays the chunks along dim 1 via
            # narrow and cat, then multiplies by B.
            A_full = all_gather_tensor(A_shard, gather_dim=0, group=group)
            chunk = A_shard.shape[0]
            A = torch.cat(
                [
                    A_full.narrow(0, 0, chunk),
                    A_full.narrow(0, chunk, chunk),
                ],
                dim=1,
            )
            return A @ B

        A_shard = torch.rand(64, 2048, device="cuda")
        torch._dynamo.decorators.mark_unbacked(
            A_shard, 0, hint_override=A_shard.shape[0]
        )
        B = torch.rand(4096, 16, device="cuda")

        gm = _make_post_grad_fx(func, A_shard, B)
        with _test_mode():
            micro_pipeline_tp_pass(gm.graph)

        self.assertIn("fused_all_gather_matmul", str(gm.graph))
        self.assertNotIn("all_gather_into_tensor", str(gm.graph))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_fuse_all_gather_matmul_slice_cat_trim(self):
        # [FUNCTION DOC] Like the slice/cat test but the concatenated result is trimmed with a
        # narrow on dim 1 before the matmul; the pass must still produce the fused op and
        # remove the plain all-gather from the graph.
        group = dist.group.WORLD

        def func(A_shard: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Gathers on dim 0, builds a dim-1 layout via narrow and cat,
            # keeps the first 4096 columns and multiplies by B.
            A_full = all_gather_tensor(A_shard, gather_dim=0, group=group)
            chunk = A_shard.shape[0]
            A = torch.cat(
                [
                    A_full.narrow(0, 0, chunk),
                    A_full.narrow(0, chunk, chunk),
                ],
                dim=1,
            )
            return A.narrow(1, 0, 4096) @ B

        A_shard = torch.rand(64, 2048, device="cuda")
        torch._dynamo.decorators.mark_unbacked(
            A_shard, 0, hint_override=A_shard.shape[0]
        )
        B = torch.rand(4096, 16, device="cuda")

        gm = _make_post_grad_fx(func, A_shard, B)
        with _test_mode():
            micro_pipeline_tp_pass(gm.graph)

        self.assertIn("fused_all_gather_matmul", str(gm.graph))
        self.assertNotIn("all_gather_into_tensor", str(gm.graph))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "Test requires FP8 support")
    @parametrize("A_dims", [2, 3])
    @parametrize("gather_dim", [0, 1, 2])
    @parametrize("return_A", [True, False])
    @fresh_cache()
    def test_fuse_all_gather_scaled_matmul(self, A_dims, gather_dim, return_A):
        # [FUNCTION DOC] Needs fp8 support. Parametrized like the unscaled test, with an fp8
        # gathered input and scaled matmul. Runs the pass on a traced graph and then on the
        # compiled code; unless gathering on the contraction dim, the fused scaled op must
        # appear and the plain all-gather must be gone, otherwise the plain all-gather stays.
        if gather_dim >= A_dims:
            return

        group = dist.group.WORLD

        def func(
            A_shard: torch.Tensor,
            B: torch.Tensor,
            A_scale: torch.Tensor,
            B_scale: torch.Tensor,
            out_dtype: torch.dtype | None,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: fp8 all-gathers A_shard, applies a scaled matmul (flattening
            # leading dims for 3D input and restoring them), returns (A or None, product).
            A = _fp8_all_gather(
                A_shard, gather_dim=gather_dim, group_name=group.group_name
            )
            if len(A_shard.shape) > 2:
                C = torch._scaled_mm(
                    A.flatten(0, -2), B, A_scale, B_scale, out_dtype=out_dtype
                )
                C = C.view(*A.shape[:-1], -1)
            else:
                C = torch._scaled_mm(A, B, A_scale, B_scale, out_dtype=out_dtype)

            if return_A:
                return A, C
            else:
                return None, C

        if A_dims == 2:
            A_shard_shape = [64, 32]
        elif A_dims == 3:
            A_shard_shape = [2, 64, 32]
        else:
            raise AssertionError(f"Invalid A_dims: {A_dims}")

        A_shard_shape[gather_dim] //= self.world_size
        A_shard = torch.rand(*A_shard_shape, device="cuda").to(e4m3_type)
        B = torch.rand(16, 32, device="cuda").to(e4m3_type).T
        A_scale = torch.tensor(0.1, device="cuda")
        B_scale = torch.tensor(0.1, device="cuda")

        gm = _make_post_grad_fx(func, A_shard, B, A_scale, B_scale, torch.bfloat16)
        with _test_mode():
            micro_pipeline_tp_pass(gm.graph)
        if gather_dim == A_dims - 1:
            self.assertNotIn("fused_all_gather_scaled_matmul", str(gm.graph))
            self.assertIn("all_gather_into_tensor", str(gm.graph))
        else:
            # Decomposing the matmul on the K dimension is not supported
            self.assertIn("fused_all_gather_scaled_matmul", str(gm.graph))
            self.assertNotIn("all_gather_into_tensor", str(gm.graph))

        with _test_mode():
            compiled = torch.compile(func)
            code = run_and_get_triton_code(
                compiled, A_shard, B, A_scale, B_scale, torch.bfloat16
            )
        if gather_dim == A_dims - 1:
            self.assertNotIn("fused_all_gather_scaled_matmul", code)
            self.assertIn("all_gather_into_tensor", code)
        else:
            # Decomposing the matmul on the K dimension is not supported
            self.assertIn("fused_all_gather_scaled_matmul", code)
            self.assertNotIn("all_gather_into_tensor", code)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @parametrize("A_dims", [2, 3])
    @parametrize("scatter_dim", [0, 1, 2])
    @fresh_cache()
    def test_fuse_matmul_reduce_scatter(self, A_dims, scatter_dim):
        # [FUNCTION DOC] Parametrized over input rank and scatter dim (out-of-range skipped).
        # Compiles matmul followed by an average reduce-scatter under symmetric-memory test
        # mode and asserts the code uses the fused matmul-reduce-scatter op with no plain
        # reduce-scatter.
        if scatter_dim >= A_dims:
            return

        group = dist.group.WORLD

        def func(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the average reduce-scatter of A times B.
            return reduce_scatter_single(A @ B, "avg", scatter_dim, group)

        if A_dims == 2:
            A = torch.rand(64, 32, device="cuda")
        elif A_dims == 3:
            A = torch.rand(2, 64, 32, device="cuda")
        else:
            raise AssertionError(f"Invalid A_dims: {A_dims}")
        B = torch.rand(32, 16, device="cuda")

        with _test_mode():
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, A, B)

        self.assertIn("fused_matmul_reduce_scatter", code)
        self.assertNotIn("reduce_scatter_tensor", code)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    def test_fuse_matmul_reduce_scatter_slice_cat(self):
        # [FUNCTION DOC] Builds a post-grad graph where the matmul output columns are halved
        # and restacked on dim 0 (narrow plus cat, with an unbacked dim) before a reduce-scatter;
        # after the pass the graph must have the fused op and no plain reduce-scatter.
        group = dist.group.WORLD

        def func(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Multiplies A by B, restacks the two column halves along
            # dim 0 and returns the average reduce-scatter on dim 0.
            C = A @ B
            half_out = B.shape[1] // 2
            C = torch.cat(
                [
                    C.narrow(1, 0, half_out),
                    C.narrow(1, half_out, half_out),
                ],
                dim=0,
            )
            return reduce_scatter_tensor(C, "avg", 0, group)

        A = torch.rand(64, 32, device="cuda")
        B = torch.rand(32, 16, device="cuda")
        torch._dynamo.decorators.mark_unbacked(B, 1, hint_override=B.shape[1])

        gm = _make_post_grad_fx(func, A, B)
        with _test_mode():
            micro_pipeline_tp_pass(gm.graph)

        self.assertIn("fused_matmul_reduce_scatter", str(gm.graph))
        self.assertNotIn("reduce_scatter_tensor", str(gm.graph))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "Test requires FP8 support")
    @parametrize("A_dims", [2, 3])
    @parametrize("scatter_dim", [0, 1, 2])
    @fresh_cache()
    def test_fuse_scaled_matmul_reduce_scatter(self, A_dims, scatter_dim):
        # [FUNCTION DOC] Needs fp8 support. Parametrized over input rank and scatter dim; the
        # last dim and beyond are skipped. Checks both a traced graph after the pass and the
        # compiled code contain the fused scaled matmul-reduce-scatter op and no plain
        # reduce-scatter.
        if scatter_dim >= A_dims - 1:
            return

        group = dist.group.WORLD

        def func(
            A: torch.Tensor,
            B: torch.Tensor,
            A_scale: torch.Tensor,
            B_scale: torch.Tensor,
            out_dtype: torch.dtype,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Scaled fp8 matmul (flattening leading dims for 3D input
            # and restoring them) followed by an average reduce-scatter on scatter_dim.
            if len(A.shape) > 2:
                C = torch._scaled_mm(
                    A.flatten(0, -2), B, A_scale, B_scale, out_dtype=out_dtype
                )
                C = C.view(*A.shape[:-1], B.shape[1])
            else:
                C = torch._scaled_mm(A, B, A_scale, B_scale, out_dtype=out_dtype)
            return reduce_scatter_single(C, "avg", scatter_dim, group)

        if A_dims == 2:
            A = torch.rand(64, 32, device="cuda").to(e4m3_type)
        elif A_dims == 3:
            A = torch.rand(2, 64, 32, device="cuda").to(e4m3_type)
        else:
            raise AssertionError(f"Invalid A_dims: {A_dims}")
        B = torch.rand(16, 32, device="cuda").to(e4m3_type).T
        A_scale = torch.tensor(0.1, device="cuda")
        B_scale = torch.tensor(0.1, device="cuda")

        gm = _make_post_grad_fx(func, A, B, A_scale, B_scale, torch.bfloat16)
        with _test_mode():
            micro_pipeline_tp_pass(gm.graph)
        self.assertIn("fused_scaled_matmul_reduce_scatter", str(gm.graph))
        self.assertNotIn("reduce_scatter_tensor", str(gm.graph))

        with _test_mode():
            compiled = torch.compile(func)
            code = run_and_get_triton_code(
                compiled, A, B, A_scale, B_scale, torch.bfloat16
            )
        self.assertIn("fused_scaled_matmul_reduce_scatter", code)
        self.assertNotIn("reduce_scatter_tensor", code)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "Test requires FP8 support")
    @parametrize("scatter_dim", [0, 1])
    @fresh_cache()
    def test_fuse_scaled_matmul_reduce_scatter_rowwise_scales_reshape_mm_reshape(
        self, scatter_dim
    ):
        # [FUNCTION DOC] Needs fp8 support. Uses rowwise scales with a reshape, scaled matmul,
        # reshape pattern followed by a sum reduce-scatter on dim 0 or 1; the pass must fuse it
        # in the traced graph. On devices with capability below 8.9 it stops there; otherwise
        # the compiled code must also contain the fused op and no plain reduce-scatter.
        group = dist.group.WORLD

        def reshape_mm_reshape(
            A: torch.Tensor,
            B: torch.Tensor,
            A_scale: torch.Tensor,
            B_scale: torch.Tensor,
            out_dtype: torch.dtype,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Flattens A and its scale to 2D, inverts the scale, runs a
            # scaled matmul, restores the leading dims of the output and returns its sum
            # reduce-scatter on scatter_dim.
            """
            Performs a scaled_mm followed by a reduce scatter,
            following the reshape -> scaled_mm -> reshape pattern.
            """
            orig_shape = A.shape

            # reshape tensor and scale together
            A = A.reshape(-1, orig_shape[-1])
            A_scale = A_scale.reshape(-1, A_scale.shape[-1])
            A_scale = torch.reciprocal(A_scale)

            C = torch._scaled_mm(A, B, A_scale, B_scale, out_dtype=out_dtype)

            # reshape output to have same leading dims as original `A` tensor
            C = C.view(*orig_shape[:-1], C.shape[-1])
            return reduce_scatter_single(C, "sum", scatter_dim, group)

        A = torch.rand(2, 16, 32, device="cuda").to(e4m3_type)
        B = torch.rand(64, 32, device="cuda").to(e4m3_type).T

        # A_scale = rowwise scales
        A_scale = torch.full((2, 16, 1), 0.1, device="cuda")

        # B_scale = rowwise scales transposed for A @ B^T
        B_scale = torch.full((1, 64), 0.1, device="cuda")

        gm = _make_post_grad_fx(
            reshape_mm_reshape, A, B, A_scale, B_scale, torch.bfloat16
        )

        with _test_mode():
            micro_pipeline_tp_pass(gm.graph)

        self.assertIn("fused_scaled_matmul_reduce_scatter", str(gm.graph))
        self.assertNotIn("reduce_scatter_tensor", str(gm.graph))

        if torch.cuda.get_device_capability() < (8, 9):
            return

        with _test_mode():
            compiled = torch.compile(reshape_mm_reshape)
            code = run_and_get_triton_code(
                compiled, A, B, A_scale, B_scale, torch.bfloat16
            )
        self.assertIn("fused_scaled_matmul_reduce_scatter", code)
        self.assertNotIn("reduce_scatter_tensor", code)

    @unittest.skipIf(
        TEST_WITH_TORCHINDUCTOR or IS_LINUX or TEST_WITH_ROCM,
        "https://github.com/pytorch/pytorch/issues/153223",
    )
    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/145924")
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @parametrize("shard_dim", [0, 1])
    @fresh_cache()
    def test_dtensor_seq_par(self, shard_dim: int):
        # [FUNCTION DOC] Makes a bias-free MLP sequence-parallel with a column-wise first layer
        # and row-wise second layer sharded on shard_dim (0 or 1), compiles it under
        # symmetric-memory test mode and asserts the code has both fused ops and no plain
        # collectives.
        # KNOWN ISSUE: one skip decorator skips on all Linux machines, so the test is
        # effectively disabled there.
        model: torch.nn.Module = MLPModule(device="cuda", bias=False)
        device_mesh = DeviceMesh(
            "cuda",
            torch.arange(0, self.world_size),
        )
        parallelize_plan = {
            "net1": ColwiseParallel(input_layouts=Shard(shard_dim)),
            "net2": RowwiseParallel(output_layouts=Shard(shard_dim)),
        }
        model = parallelize_module(model, device_mesh, parallelize_plan)
        if shard_dim == 0:
            inp = torch.rand(8, 10, device="cuda")
        elif shard_dim == 1:
            inp = torch.rand(2, 8, 10, device="cuda")
        else:
            raise AssertionError("Invalid shard_dim")

        with _test_mode():
            compiled = torch.compile(model)
            code = run_and_get_triton_code(compiled, inp)

        self.assertIn("fused_all_gather_matmul", code)
        self.assertNotIn("all_gather_into_tensor", code)
        self.assertIn("fused_matmul_reduce_scatter", code)
        self.assertNotIn("reduce_scatter_tensor", code)


@instantiate_parametrized_tests
class MicroPipelineTP4GPUTest(TestCase):
    # [CLASS DOC] Same fixture as MicroPipelineTPTest but with a fake 4-rank world, used for
    # tests needing a 2D device mesh.
    def setUp(self):
        # [FUNCTION DOC] Enables the global Inductor micro-pipeline flag, sets rank 0 and world
        # size 4, selects cuda:0 and initializes a fake-backend default process group.
        super().setUp()
        torch._inductor.config._micro_pipeline_tp = True

        self.rank = 0
        self.world_size = 4
        torch.cuda.set_device("cuda:0")

        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    def tearDown(self):
        # [FUNCTION DOC] Destroys the default process group created in setUp.
        # KNOWN ISSUE: does not call the parent tearDown or reset the Inductor flag.
        dist.destroy_process_group()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @fresh_cache()
    @torch._inductor.config.patch(shape_padding=False)
    def test_extra_collectives(self):
        # [FUNCTION DOC] On a 2x2 mesh with dims tp and other, compiles a function with
        # collectives on both mesh dims and only the tp group enabled for symmetric memory.
        # The tp all-gather/reduce-scatter around matmuls must be fused, while collectives on the
        # other dim stay as plain ones; the code is checked for both kinds of op.
        device_mesh = DeviceMesh(
            "cuda",
            torch.arange(0, self.world_size).view(2, -1),
            mesh_dim_names=("tp", "other"),
        )

        def func(inp: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Gathers inp over the tp dim and applies w1, gathers over the
            # other dim, normalizes in place, reduce-scatters over the other dim, applies w2 and
            # reduce-scatters over the tp dim; returns the final tensor.
            hidden = all_gather_single(inp, 0, (device_mesh, 0)) @ w1.t()
            full_hidden = all_gather_single(hidden, 0, (device_mesh, 1))
            full_hidden /= full_hidden.pow(2).sum().sqrt()
            hidden = reduce_scatter_single(full_hidden, "avg", 0, (device_mesh, 1))
            return reduce_scatter_single(hidden @ w2.t(), "avg", 0, (device_mesh, 0))

        inp = torch.rand(8, 10, device="cuda")
        w1 = torch.rand(7, 10, device="cuda")
        w2 = torch.rand(10, 7, device="cuda")

        with _test_mode(group_names={device_mesh["tp"].get_group().group_name}):
            compiled = torch.compile(func)
            code = run_and_get_triton_code(compiled, inp, w1, w2)

        self.assertIn("fused_all_gather_matmul", code)
        self.assertIn("all_gather_into_tensor", code)
        self.assertIn("fused_matmul_reduce_scatter", code)
        self.assertIn("reduce_scatter_tensor", code)


if __name__ == "__main__":
    run_tests()
