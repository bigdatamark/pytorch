# Owner(s): ["module: inductor"]
# [MODULE DOC] Unit tests for the decomp_gram_matrix_all_gather FX pass from inductor's
# decomp_comms. Each test traces a function that all-gathers a shard and computes Gram-style
# matmuls (Muon or Shampoo like patterns), using fake tensors and make_fx, then runs the pass
# and counts all_gather, all_reduce and split nodes in the graph. Import time: defines aten and
# c10d functional op namespaces; the tests need an accelerator with a GPU build and an nccl or
# xccl backend.
"""
Unit tests for decomp_gram_matrix_all_gather FX pass.

Verifies the pass transforms:
    all_gather(X_shard) -> wait -> slice -> [Gram compute] -> split -> getitem
into:
    X_shard -> [local compute + all_reduce for Gram aggregation]
"""

import operator
import unittest

import torch
import torch.distributed as dist
import torch.fx as fx
from torch._inductor.fx_passes.decomp_comms import decomp_gram_matrix_all_gather
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import requires_accelerator_dist_backend
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.inductor_utils import HAS_GPU


aten = torch.ops.aten
c10d = torch.ops._c10d_functional


def _count_ops(graph: fx.Graph, target) -> int:  # type: ignore[type-arg]
    # [FUNCTION DOC] HELPER: Returns the number of call_function nodes in the FX graph whose target
    # is exactly the given op.
    return sum(1 for n in graph.nodes if n.op == "call_function" and n.target is target)


@requires_accelerator_dist_backend(["nccl", "xccl"])
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestDecompGramMatrixAllGather(InductorTestCase):
    # [CLASS DOC] Tests for the decomp_gram_matrix_all_gather pass. The class initializes a fake
    # process group of world size 2 (rank 0) once and tears it down at the end; tests use the
    # group name 0 and device cuda. Covers positive transformations and cases where the pass
    # must leave the graph unchanged.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class-level setup: calls the base setup, creates a fake process group
        # with a FakeStore (rank 0, world size 2) and records cuda as the test device on the
        # class. Side effect: initializes the default process group.
        super().setUpClass()
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        cls.device = "cuda"

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class-level teardown: calls the base teardown and destroys the default
        # process group.
        super().tearDownClass()
        dist.destroy_process_group()

    def test_muon_newton_schulz(self):
        # [FUNCTION DOC] Traces a Muon-style Newton-Schulz step over a gathered shard (64 by 32,
        # gathered 128 by 32, transposed before the Gram matmuls) and asserts that the pass
        # removes the all_gather and the split, and introduces at least ns_steps all_reduce
        # nodes (one per Gram matmul, plus norm reduction).
        """
        Rigi Muon optimizer pattern with initial transpose (rows > cols).

        Shard (S, K) where S*W > K. After all_gather: (S*W, K).
        Muon transposes to (K, S*W), then Gram = X @ X.T = (K, K).
        The Gram output (K, K) does NOT depend on gathered dim (S*W),
        so the decomposition is shape-correct.

        After transpose back and scale, result is (S*W, K), split to (S, K).
        """
        group_name = "0"
        world_size = 2
        ns_steps = 5

        def muon_step(x_shard):
            # [FUNCTION DOC] HELPER: Traced function: all-gathers the shard, waits, applies an
            # identity slice, normalizes in bfloat16, transposes, runs the iterative Newton-Schulz
            # matmul updates, transposes back, scales, splits by the shard size and returns the
            # first chunk.
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            # Identity slice — keeps full gathered tensor
            X = aten.slice.Tensor(waited, 0, 0, x_shard.shape[0] * world_size)

            X = aten._to_copy.default(X, dtype=torch.bfloat16)
            norm_val = aten.linalg_vector_norm.default(X, 2, [-2, -1], True)
            norm_val = aten.clamp_min.default(norm_val, 1e-7)
            X = aten.div.Tensor(X, norm_val)

            # Transpose: (S*W, K) -> (K, S*W), Gram will be (K, K)
            X = aten.transpose.int(X, -2, -1)

            for _ in range(ns_steps):
                A = aten.mm.default(X, aten.transpose.int(X, -2, -1))
                B = aten.add.Tensor(
                    aten.mul.Tensor(A, -4.7750),
                    aten.mul.Tensor(aten.mm.default(A, A), 2.0315),
                )
                X = aten.add.Tensor(
                    aten.mul.Tensor(X, 3.4445),
                    aten.mm.default(B, X),
                )

            # Transpose back: (K, S*W) -> (S*W, K)
            X = aten.transpose.int(X, -2, -1)
            X = aten.mul.Tensor(X, 0.2 * (512**0.5))

            # Split back to shard
            chunks = aten.split.Tensor(X, x_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            # S=64, K=32. Gathered=(128,32). After transpose X=(32,128).
            # Gram=(32,32). gathered_dim=128, gram_dim=32 → decomposable.
            x_shard = torch.randn(64, 32, device=self.device)
            traced = make_fx(muon_step)(x_shard)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 0
        )
        # 5 all_reduces for Gram mms + 1 for the norm reduction
        self.assertGreaterEqual(
            _count_ops(traced.graph, c10d.all_reduce.default), ns_steps
        )
        self.assertEqual(_count_ops(traced.graph, aten.split.Tensor), 0)

    def test_no_slice_gathered_consumed_directly(self):
        # [FUNCTION DOC] Traces a variant where the wait output feeds the computation directly
        # (no slice, and multiple users of the wait: norm and divide). Asserts the precondition
        # of no slice nodes, then that the pass still removes the all_gather and split and adds
        # all_reduce nodes.
        """
        Resilient anchor: when inductor folds the FSDP no-op identity slice
        (even sharding), the gathered tensor is consumed straight off the
        wait, with no aten.slice. The pass must still anchor on
        wait(all_gather) and transform. The wait here also has multiple direct
        users (norm + div), which the previous wait.users==1 gate rejected.
        """
        group_name = "0"
        world_size = 2
        ns_steps = 3

        def muon_no_slice(x_shard):
            # [FUNCTION DOC] HELPER: Traced function: all-gathers and waits, normalizes the waited
            # tensor directly without a slice, transposes, runs Gram-style matmul updates,
            # transposes back, splits by the shard size and returns the first chunk.
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            # No slice: wait feeds the chain directly (norm + div).
            norm_val = aten.linalg_vector_norm.default(waited, 2, [-2, -1], True)
            norm_val = aten.clamp_min.default(norm_val, 1e-7)
            X = aten.div.Tensor(waited, norm_val)

            X = aten.transpose.int(X, -2, -1)
            for _ in range(ns_steps):
                A = aten.mm.default(X, aten.transpose.int(X, -2, -1))
                X = aten.add.Tensor(X, aten.mm.default(A, X))
            X = aten.transpose.int(X, -2, -1)

            chunks = aten.split.Tensor(X, x_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            x_shard = torch.randn(64, 32, device=self.device)
            traced = make_fx(muon_no_slice)(x_shard)

        # Precondition: the gathered tensor is consumed with no slice.
        self.assertEqual(_count_ops(traced.graph, aten.slice.Tensor), 0)
        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
        )

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default),
            0,
            "all_gather should be eliminated even with no slice wrapping the wait",
        )
        self.assertGreaterEqual(
            _count_ops(traced.graph, c10d.all_reduce.default), ns_steps
        )
        self.assertEqual(_count_ops(traced.graph, aten.split.Tensor), 0)

    def test_no_transform_unsafe_entry_region(self):
        # [FUNCTION DOC] Builds three traced graphs where the entry region is unsafe (the waited
        # tensor has a side user, a column-dimension slice, or an unsupported amax reduction)
        # and asserts that after the pass the all_gather is still present and no all_reduce was
        # added.
        group_name = "0"
        world_size = 2
        ns_steps = 3

        def finish(X, x_shard):
            # [FUNCTION DOC] HELPER: Shared tail of the traced functions: transposes, runs
            # matmul updates for ns_steps iterations, transposes back, splits by the shard
            # size and returns the first chunk.
            X = aten.transpose.int(X, -2, -1)
            for _ in range(ns_steps):
                A = aten.mm.default(X, aten.transpose.int(X, -2, -1))
                X = aten.add.Tensor(X, aten.mm.default(A, X))
            X = aten.transpose.int(X, -2, -1)
            return operator.getitem(aten.split.Tensor(X, x_shard.shape[0], 0), 0)

        def side_user(x_shard):
            # [FUNCTION DOC] HELPER: Traced function whose waited gathered tensor has an additional
            # side consumer (a multiply) besides the main chain; returns the main result and the
            # side value.
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            side = aten.mul.Tensor(waited, 2.0)
            return finish(waited, x_shard), side

        def column_slice(x_shard):
            # [FUNCTION DOC] HELPER: Traced function that slices the gathered tensor along
            # dimension 1 (columns 0 to 16) before the shared tail computation.
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            return finish(aten.slice.Tensor(waited, 1, 0, 16), x_shard)

        def unsupported_reduction(x_shard):
            # [FUNCTION DOC] HELPER: Traced function that subtracts a per-column amax
            # reduction over dimension 0 of the waited tensor before the shared tail
            # computation.
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            return finish(
                aten.sub.Tensor(waited, aten.amax.default(waited, [0], True)), x_shard
            )

        for fn in (side_user, column_slice, unsupported_reduction):
            with FakeTensorMode():
                traced = make_fx(fn)(torch.randn(64, 32, device=self.device))

            decomp_gram_matrix_all_gather(traced)

            self.assertEqual(
                _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
            )
            self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)

    def test_shampoo_preconditioner(self):
        # [FUNCTION DOC] Traces a Shampoo-like iterative preconditioner (3 iterations of
        # permute-based Gram matmul) over the gathered shard. Asserts that the pass removes the
        # all_gather and split and produces exactly one all_reduce per iteration.
        """
        Shampoo-like optimizer with iterative preconditioner (3 iterations).
        G.T @ G right Kronecker factor via aten.permute.
        Multiple Gram mms per all_gather to pass the NS-pattern threshold.
        """
        group_name = "0"
        world_size = 2
        n_iters = 3

        def shampoo_step(g_shard):
            # [FUNCTION DOC] HELPER: Traced function: all-gathers and waits, slices the full
            # range, runs the coupled iteration with a permuted transpose Gram matmul each
            # step, multiplies by the permuted result, splits by the shard size and returns
            # the first chunk.
            gathered = c10d.all_gather_into_tensor.default(
                g_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            g = aten.slice.Tensor(waited, 0, 0, g_shard.shape[0] * world_size)

            # Iterative preconditioner (coupled Newton for matrix sqrt)
            precond = g
            for _ in range(n_iters):
                gt = aten.permute.default(precond, [1, 0])
                gram = aten.mm.default(gt, precond)  # G.T @ G = (K, K)
                precond = aten.sub.Tensor(precond, aten.mm.default(precond, gram))

            g_precond = aten.mm.default(g, aten.permute.default(precond, [1, 0]))

            chunks = aten.split.Tensor(g_precond, g_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            g_shard = torch.randn(64, 32, device=self.device)
            traced = make_fx(shampoo_step)(g_shard)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
        )

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 0
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), n_iters)
        self.assertEqual(_count_ops(traced.graph, aten.split.Tensor), 0)

    def test_no_transform_without_gram(self):
        # [FUNCTION DOC] Traces a graph with a matmul against an independent weight (not a Gram
        # pattern) and asserts the pass leaves the all_gather and adds no all_reduce.
        """
        Pass must NOT transform when no Gram mm is detected.
        mm(X, W) with independent W is not a Gram pattern.
        """
        group_name = "0"
        world_size = 2

        def no_gram(x_shard, weight):
            # [FUNCTION DOC] HELPER: Traced function taking a shard and a weight: all-gathers,
            # waits, slices, multiplies by the weight, splits by the shard size and returns the
            # first chunk.
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            x = aten.slice.Tensor(waited, 0, 0, x_shard.shape[0] * world_size)

            out = aten.mm.default(x, weight)

            chunks = aten.split.Tensor(out, x_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            x_shard = torch.randn(32, 64, device=self.device)
            weight = torch.randn(64, 16, device=self.device)
            traced = make_fx(no_gram)(x_shard, weight)

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default),
            1,
            "all_gather should remain when no Gram pattern detected",
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)

    def test_no_transform_non_decomposable_shape(self):
        # [FUNCTION DOC] Traces Gram matmuls on a square gathered tensor (64 by 64, so the Gram
        # output size depends on the gathered dimension) and asserts the pass does not
        # transform: the all_gather remains and no all_reduce appears.
        """
        mm(X, X.T) where X is (N, N) square: Gram output (N, N) depends
        on the gathered dim, so the decomposition is NOT shape-correct.
        Pass must not transform.
        """
        group_name = "0"
        world_size = 2
        ns_steps = 3

        def square_gram(x_shard):
            # [FUNCTION DOC] HELPER: Traced function: all-gathers, waits, slices, runs matmul
            # updates with a square Gram product for ns_steps iterations, splits by the shard size
            # and returns the first chunk.
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            X = aten.slice.Tensor(waited, 0, 0, x_shard.shape[0] * world_size)

            for _ in range(ns_steps):
                # mm(X, X.T): output is (N, N) where N = gathered rows
                A = aten.mm.default(X, aten.transpose.int(X, -2, -1))
                X = aten.add.Tensor(X, aten.mm.default(A, X))

            chunks = aten.split.Tensor(X, x_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            # Shard (32, 64), gathered (64, 64) -- square.
            # Gram = mm(X, X.T) = (64, 64), depends on gathered dim.
            x_shard = torch.randn(32, 64, device=self.device)
            traced = make_fx(square_gram)(x_shard)

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default),
            1,
            "all_gather should remain for non-decomposable Gram shape",
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)

    def test_no_transform_single_gram_mm(self):
        # [FUNCTION DOC] Traces a graph with a single Gram matmul (below the minimum count of 2
        # named in the docstring) and asserts that the pass leaves the all_gather in place and
        # adds no all_reduce.
        """
        Single Gram mm doesn't meet the _MIN_GRAM_MMS=2 threshold.
        Pass must not transform (likely forward/backward, not iterative optimizer).
        """
        group_name = "0"
        world_size = 2

        def single_gram(x_shard):
            # [FUNCTION DOC] HELPER: Traced function: all-gathers, waits, slices, computes one Gram
            # matmul and a product with it, splits by the shard size and returns the first chunk.
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            X = aten.slice.Tensor(waited, 0, 0, x_shard.shape[0] * world_size)

            Xt = aten.transpose.int(X, -2, -1)
            gram = aten.mm.default(Xt, X)
            out = aten.mm.default(X, gram)

            chunks = aten.split.Tensor(out, x_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            x_shard = torch.randn(64, 32, device=self.device)
            traced = make_fx(single_gram)(x_shard)

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default),
            1,
            "all_gather should remain with only 1 Gram mm (below threshold)",
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)

    def test_padded_shard_trimming(self):
        # [FUNCTION DOC] Traces an FSDP-padded case (padded shard of 64 rows, 60 unpadded rows
        # per rank, slice to 120 rows). Asserts the pass removes the all_gather, adds all_reduce
        # nodes, and inserts exactly one slice on the placeholder whose end argument is 60,
        # trimming the shard.
        """
        FSDP padding: the shard is padded for divisibility, but the gathered
        tensor is sliced down to the unpadded total before the Gram compute
        and split into unpadded per-rank chunks. The getitem output
        (unpadded_per_rank) is then smaller than the padded shard, so the
        pass must insert a slice trimming the shard down to the unpadded size.
        """
        group_name = "0"
        world_size = 2
        ns_steps = 3
        shard_size = 64  # padded shard
        unpadded_per_rank = 60
        gathered_unpadded = unpadded_per_rank * world_size

        def padded_step(x_shard):
            # [FUNCTION DOC] HELPER: Traced function: all-gathers and waits, slices to the unpadded
            # total row count, transposes, runs matmul updates, transposes back, splits into
            # unpadded per-rank chunks and returns the first.
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            # Slice strips padding: keep only the unpadded total rows.
            X = aten.slice.Tensor(waited, 0, 0, gathered_unpadded)

            X = aten.transpose.int(X, -2, -1)

            for _ in range(ns_steps):
                A = aten.mm.default(X, aten.transpose.int(X, -2, -1))
                X = aten.add.Tensor(X, aten.mm.default(A, X))

            X = aten.transpose.int(X, -2, -1)

            # Even split into unpadded per-rank chunks (60 < padded 64).
            chunks = aten.split.Tensor(X, unpadded_per_rank, 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            x_shard = torch.randn(shard_size, 32, device=self.device)
            traced = make_fx(padded_step)(x_shard)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
        )

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 0
        )
        self.assertGreaterEqual(
            _count_ops(traced.graph, c10d.all_reduce.default), ns_steps
        )
        # Trim path under test: pass slices the placeholder shard to unpadded size.
        trim_slices = [
            n
            for n in traced.graph.nodes
            if n.op == "call_function"
            and n.target is aten.slice.Tensor
            and isinstance(n.args[0], fx.Node)
            and n.args[0].op == "placeholder"
            and len(n.args) >= 4
            and n.args[3] == unpadded_per_rank
        ]
        self.assertEqual(len(trim_slices), 1, "pass should insert a shard-trim slice")

    def test_no_transform_multiple_consumed_getitems(self):
        # [FUNCTION DOC] Traces a graph where two getitems of the split are both consumed, and
        # asserts the pass bails out: the all_gather remains and no all_reduce is added.
        """
        When the split feeds multiple consumed getitems (e.g. several ranks'
        shards used in the same graph), collapsing the split into a single
        rank-local result would leave the sibling getitems reading past the
        now shard-sized split input. The pass must bail rather than corrupt
        the graph.
        """
        group_name = "0"
        world_size = 2
        ns_steps = 3

        def multi_user_step(x_shard):
            # [FUNCTION DOC] HELPER: Traced function: gather, wait, slice, transpose, matmul
            # updates, transpose back, split by the shard size, then returns the sum of the first
            # two chunks.
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            X = aten.slice.Tensor(waited, 0, 0, x_shard.shape[0] * world_size)

            X = aten.transpose.int(X, -2, -1)

            for _ in range(ns_steps):
                A = aten.mm.default(X, aten.transpose.int(X, -2, -1))
                X = aten.add.Tensor(X, aten.mm.default(A, X))

            X = aten.transpose.int(X, -2, -1)

            chunks = aten.split.Tensor(X, x_shard.shape[0], 0)
            # Both getitems have users
            c0 = operator.getitem(chunks, 0)
            c1 = operator.getitem(chunks, 1)
            return aten.add.Tensor(c0, c1)

        with FakeTensorMode():
            x_shard = torch.randn(64, 32, device=self.device)
            traced = make_fx(multi_user_step)(x_shard)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
        )

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default),
            1,
            "all_gather should remain when split has other consumed getitems",
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)


if __name__ == "__main__":
    run_tests()
