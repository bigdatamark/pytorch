# Owner(s): ["oncall: pt2"]
# [MODULE DOC] Tests for the activation memory budget feature of the partitioner (config option
# activation_memory_budget). Helpers measure CUDA activation memory in MiB and backward FLOPs
# (normalized so a 512 square matmul is 1) for a function compiled with the
# aot_eager_decomp_partition backend at a given budget; tests check the memory and recompute
# choices. Imports triton only if available. The main block runs tests only when CUDA and triton
# exist and ROCm is not used.
import random
import unittest
from math import prod

import torch
import torch._functorch.config as config
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.utils._triton import has_triton
from torch.utils.checkpoint import checkpoint
from torch.utils.flop_counter import FlopCounterMode, register_flop_formula


if has_triton():
    # note: if we only import triton in the test, the test fails:
    # def relu_kernel_(inp_ptr, out_ptr, sz, BLOCK_SIZE: tl.constexpr):
    # NameError('tl is not defined')
    import triton
    import triton.language as tl


def compile_with_ac(f, memory_budget):
    # [FUNCTION DOC] HELPER: Returns the function f compiled with the aot_eager_decomp_partition
    # backend. KNOWN ISSUE: the memory_budget argument is accepted but never used; the function
    # is also not called anywhere in this file.
    return torch.compile(f, backend="aot_eager_decomp_partition")


def get_act_mem(f):
    # [FUNCTION DOC] HELPER: Measures activation memory of a zero-argument function returning a
    # scalar. Runs it once with backward as warmup, records requested CUDA bytes, runs the
    # forward again, and returns the growth in requested bytes in MiB. Then runs backward to
    # release the graph. Requires CUDA.
    out = f()
    out.backward()
    start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
    out = f()
    cur_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
    act_mem = (cur_mem - start_mem) / (1024 * 1024)
    out.backward()
    return act_mem


def get_bw_flops(f):
    # [FUNCTION DOC] HELPER: Measures backward FLOPs of a zero-argument function returning a
    # scalar. Runs one warmup forward and backward, then counts FLOPs in a second backward with
    # the flop counter and returns the total divided by the FLOPs of a 512 square matmul.
    # Normalized so that a 512 square matmul returns 1
    f().backward()
    out = f()
    with FlopCounterMode(display=False) as mode:
        out.backward()
    return mode.get_total_flops() / (512**3 * 2)


def create_pair(B_I, O):
    # [FUNCTION DOC] HELPER: Creates an input of size B_I*512 square and a weight of size
    # B_I*512 by O*512, both random float tensors requiring grad. Per the comment they cost B_I
    # times O MiB of memory and B_I squared times O units of matmul FLOPs. Returns the pair.
    # results in B_I * O memory, requires B_I * B_I * O flops
    # arithmetic intensity of B_I
    x = torch.randn(B_I * 512, B_I * 512, requires_grad=True)
    w = torch.randn(B_I * 512, O * 512, requires_grad=True)
    return x, w


def get_mem_and_flops(f, memory_budget=None):
    # [FUNCTION DOC] HELPER: Resets dynamo, then under a patched activation_memory_budget config
    # optionally compiles f (only if memory_budget is not None, with the same backend as
    # above) and returns activation memory in MiB rounded to one decimal, and normalized
    # backward FLOPs.
    # Returns megabytes rounded to 1 decimal point and FLOPs
    # Note that each value of size (512, 512, torch.float32) is 1 MiB
    torch._dynamo.reset()
    with config.patch(activation_memory_budget=memory_budget):
        if memory_budget is not None:
            f = torch.compile(f, backend="aot_eager_decomp_partition")

        # We round this to nearest 10th of a megabyte.
        return round(get_act_mem(f), 1), get_bw_flops(f)


class MemoryBudgetTest(TestCase):
    # [CLASS DOC] Test case for the memory-budget based choice of what to save or recompute.
    # Every test uses the CUDA default device, set in setUp and unset in tearDown.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp and sets the default device to cuda.
        super().setUp()
        torch.set_default_device("cuda")

    def tearDown(self):
        # [FUNCTION DOC] Resets the default device to none, then calls the base tearDown.
        torch.set_default_device(None)
        super().tearDown()

    def test_rematerializes_cheap(self):
        # [FUNCTION DOC] For cosine followed by a matmul and sum on 512 square inputs: eager mem
        # is 1 MiB. With budget 1.0 memory stays 1 and FLOPs equal eager; with budget 0.5 the
        # cosine is recomputed, giving 0 MiB with unchanged FLOPs.
        def f(x, w):
            # [FUNCTION DOC] HELPER: Applies cosine, a matmul with w, and returns the sum.
            x = x.cos()
            x = torch.mm(x, w)
            return x.sum()

        x = torch.randn(512, 512, requires_grad=True)
        w = torch.randn(512, 512, requires_grad=True)

        def call():
            # [FUNCTION DOC] HELPER: Calls f with the enclosing x and w.
            return f(x, w)

        eager_mem, eager_flops = get_mem_and_flops(call)
        self.assertEqual(eager_mem, 1.0)
        mem_10, flops_10 = get_mem_and_flops(call, memory_budget=1.0)
        # Recomputing `.cos()` is not free here.
        self.assertEqual(mem_10, 1.0)
        self.assertEqual(eager_flops, flops_10)
        mem_5, flops_5 = get_mem_and_flops(call, memory_budget=0.5)
        # We can just recompute `x.cos()` here to only depend on the inputs
        self.assertEqual(mem_5, 0.0)
        self.assertEqual(flops_5, eager_flops)

    def test_matmul_even_chain(self):
        # [FUNCTION DOC] Chain of five matmuls with cosines. Loops over budgets 0.0 to 1.0 in
        # steps of 0.1. For budgets up to 0.5 expects memory equal to the budget index and FLOPs
        # of eager plus the number of unsaved matmuls; between 0.5 and 1 expects 5 MiB and eager
        # FLOPs; at 1.0 expects 10 MiB and eager FLOPs.
        def f(x, ws):
            # [FUNCTION DOC] HELPER: Applies cosine, then for each weight a matmul and cosine;
            # returns the sum.
            x = x.cos()
            for w in ws:
                x = torch.mm(x, w).cos()
            return x.sum()

        x = torch.randn(512, 512, requires_grad=True)
        ws = [torch.randn(512, 512, requires_grad=True) for _ in range(5)]

        def call():
            # [FUNCTION DOC] HELPER: Calls f with the enclosing x and ws.
            return f(x, ws)

        _, eager_flops = get_mem_and_flops(call)
        for budget in range(11):
            mem, flops = get_mem_and_flops(call, memory_budget=budget / 10)
            if budget <= 5:
                # We start saving the matmuls
                self.assertEqual(mem, budget)
                self.assertEqual(flops, eager_flops + (5 - budget))
            elif budget < 10:
                # We're only recomputing the `cos` operations
                self.assertEqual(mem, 5.0)
                self.assertEqual(flops, eager_flops)
            elif budget == 10:
                self.assertEqual(mem, 10.0)
                self.assertEqual(flops, eager_flops)

    def test_matmul_uneven_chain(self):
        # [FUNCTION DOC] Checks the knapsack solver on weights of different widths whose matmuls
        # have equal compute per saved memory. For three configs (two hand-written and one with
        # random subset sums from a seeded generator) eager memory must equal the sum of the
        # widths, and for each listed achievable memory the compiled run at the matching budget
        # fraction must hit exactly that memory.
        # This function is constructed so that we are saving one input of size
        # [512, in_dim] for each w
        # In addition, every matmul has a same ratio of compute to "memory
        # saved", so this test is essentially testing our knapsack solving

        def f(x, ws):
            # [FUNCTION DOC] HELPER: Matmul of x with each weight followed by cosine; returns
            # the sum of all outputs.
            xs = [torch.mm(x, w).cos() for w in ws]
            return sum(x.sum() for x in xs)

        x = torch.randn(512, 512, requires_grad=True)

        def make_weights(w_shapes):
            # [FUNCTION DOC] HELPER: Builds a list of random weights of shape 512 by dim*512
            # requiring grad, one per entry in the given width list.
            ws = []
            for dim in w_shapes:
                ws.append(torch.randn(512, dim * 512, requires_grad=True))
            return ws

        weight_configs = [
            (
                [11, 3, 4, 2],
                [
                    18,  # 11 + 4 + 3
                    17,  # 11 + 4 + 2
                    16,  # 11 + 3 + 2
                    15,  # 11 + 4
                    14,  # 11 + 3
                    13,  # 11 + 2
                    11,  # 11 + 2
                    7,  # 4 + 3
                    6,  # 4 + 2
                    5,  # 3 + 2
                ],
            ),
            (
                [3, 5, 11, 17, 14],
                [
                    42,  # 17 + 14 + 9
                    30,  # 11 + 15 + 5
                    19,  # 11 + 5 + 3
                    8,  # 5 + 3
                    3,  # 3
                ],
            ),
        ]
        random.seed(0)
        random_arr = [random.randint(0, 50) for _ in range(10)]
        exact_sums = []
        for i in range(10):
            random.shuffle(random_arr)
            exact_sums.append(sum(random_arr[:i]))
        weight_configs.append((random_arr, exact_sums))

        for weight_shapes, exact_solves in weight_configs:
            ws = make_weights(weight_shapes)

            def call():
                # [FUNCTION DOC] HELPER: Calls f with the enclosing x and the weights of the
                # current config.
                return f(x, ws)

            eager_mem, _ = get_mem_and_flops(call)
            total_mem = sum(weight_shapes)
            self.assertEqual(eager_mem, sum(weight_shapes))
            for mem_achieved in exact_solves:
                mem, _ = get_mem_and_flops(call, memory_budget=mem_achieved / total_mem)
                self.assertEqual(mem, mem_achieved)

    # needs CUDA, but this test file all needs CUDA.
    @unittest.skipIf(not has_triton(), "test needs triton")
    def test_custom_triton_kernel(self):
        # [FUNCTION DOC] Skipped without triton. Defines a triton relu kernel, registers forward
        # and backward custom ops under the testac namespace with an autograd formula (the
        # backward kernel is deliberately incorrect) and a FLOP formula, then compares compiled
        # output with eager for budgets 0.0 to 1.0. Only outputs are compared, not memory.
        # [FUNCTION DOC] HELPER: Triton kernel. Each program handles one block of BLOCK_SIZE
        # elements; loads values under a bounds mask against sz, clamps negatives to zero, and
        # stores to the output pointer. Written to a buffer; returns nothing.
        @triton.jit
        def relu_kernel_(inp_ptr, out_ptr, sz, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            block = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
            msk = block < sz
            inp = tl.load(inp_ptr + block, mask=msk)
            relu = tl.where(inp < 0, 0, inp)
            tl.store(out_ptr + block, relu, mask=msk)

        @torch._library.triton_op("testac::triton_relu", mutates_args=())
        def triton_relu(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op wrapping the relu kernel. Allocates an output
            # like the input, launches with a grid of ceil(size/256) blocks, returns the output.
            y = torch.empty_like(x)
            sz = y.numel()
            BLOCK_SIZE = 256
            grid = (triton.cdiv(sz, BLOCK_SIZE),)
            torch._library.capture_triton(relu_kernel_)[grid](x, y, sz, BLOCK_SIZE)
            return y

        @torch._library.triton_op("testac::triton_relu_backward", mutates_args=())
        def triton_relu_backward(grad_out: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op standing in for relu backward. It runs the same
            # relu kernel over the incoming gradient (not a correct gradient, per its comment)
            # and returns it.
            grad_x = torch.empty_like(grad_out)
            sz = grad_out.numel()
            BLOCK_SIZE = 256
            grid = (triton.cdiv(sz, BLOCK_SIZE),)
            # I know this is wrong, but whatever..
            torch._library.capture_triton(relu_kernel_)[grid](
                grad_out, grad_x, sz, BLOCK_SIZE
            )
            return grad_x

        def _triton_relu_backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Autograd backward for the relu op; ignores ctx and calls
            # the backward custom op on the gradient.
            return triton_relu_backward(grad_out)

        def _triton_relu_setup_context(ctx, inputs, output):
            # [FUNCTION DOC] HELPER: Setup-context hook for autograd registration; does nothing.
            pass

        triton_relu.register_autograd(
            _triton_relu_backward,
            setup_context=_triton_relu_setup_context,
        )

        @register_flop_formula(
            [torch.ops.testac.triton_relu, torch.ops.testac.triton_relu_backward]
        )
        def triton_relu_flops(inp_shape, *args, **kwargs):
            # [FUNCTION DOC] HELPER: FLOP formula for both relu custom ops: the product of the
            # input shape dims. Extra arguments are ignored.
            return prod(inp_shape)

        def f(x, ws):
            # [FUNCTION DOC] HELPER: Applies the relu op to x, then for each weight a matmul
            # followed by the relu op; returns the sum.
            x = torch.ops.testac.triton_relu(x)
            for w in ws:
                x = torch.ops.testac.triton_relu(torch.mm(x, w))
            return x.sum()

        x = torch.randn(512, 512, requires_grad=True, device="cuda")
        ws = [
            torch.randn(512, 512, requires_grad=True, device="cuda") for _ in range(5)
        ]

        def call():
            # [FUNCTION DOC] HELPER: Calls f with the enclosing x and ws.
            return f(x, ws)

        expected = call()
        for budget in range(11):
            memory_budget = budget / 10
            torch._dynamo.reset()
            with config.patch(activation_memory_budget=memory_budget):
                if memory_budget is not None:
                    f_compile = torch.compile(
                        call, backend="aot_eager_decomp_partition"
                    )

                self.assertEqual(expected, f_compile())

    def test_prioritize_cheaper_matmul(self):
        # [FUNCTION DOC] Two matmul pairs of different cost. Eager memory must be 8 MiB and
        # FLOPs 24; with a budget of half, memory must be 4 and the extra FLOPs 4, i.e. the
        # cheaper first matmul is recomputed.
        def f(xs, ws):
            # [FUNCTION DOC] HELPER: For each input and weight pair, matmul then cosine; returns
            # the sum.
            xs = [torch.mm(x, w).cos() for x, w in zip(xs, ws)]
            return sum(x.sum() for x in xs)

        x1, w1 = create_pair(1, 4)
        x2, w2 = create_pair(2, 2)

        def call():
            # [FUNCTION DOC] HELPER: Calls f on the list of the two inputs and the list of the
            # two weights.
            return f([x1, x2], [w1, w2])

        eager_mem, eager_flops = get_mem_and_flops(call)
        self.assertEqual(eager_mem, 8)
        self.assertEqual(eager_flops, 24)
        comp_mem, comp_flops = get_mem_and_flops(call, memory_budget=0.5)
        self.assertEqual(comp_mem, 4)
        # We are recomputing x1 @ w1 here!
        self.assertEqual(comp_flops, eager_flops + 4)

    @config.patch(activation_memory_budget_runtime_estimator="profile")
    def test_profile(self):
        # [FUNCTION DOC] Runs with the runtime estimator config set to profile. For the five-
        # matmul cosine chain with budget 0.2 expects 2 MiB memory and three extra FLOP units
        # over eager.
        def f(x, ws):
            # [FUNCTION DOC] HELPER: Cosine, then for each weight a matmul and cosine; returns
            # the sum.
            x = x.cos()
            for w in ws:
                x = torch.mm(x, w).cos()
            return x.sum()

        x = torch.randn(512, 512, requires_grad=True)
        ws = [torch.randn(512, 512, requires_grad=True) for _ in range(5)]

        def call():
            # [FUNCTION DOC] HELPER: Calls f with the enclosing x and ws.
            return f(x, ws)

        _, eager_flops = get_mem_and_flops(call)
        mem, flops = get_mem_and_flops(call, memory_budget=0.2)
        # We start saving the matmuls
        self.assertEqual(mem, 2)
        self.assertEqual(flops, eager_flops + 3)

    def test_prioritize_cheaper_matmul2(self):
        # [FUNCTION DOC] Three matmul pairs of sizes (4,4), (6,2), (2,6). Eager memory is 40 MiB
        # and FLOPs 320. At a 28 MiB budget memory must be 28 with recompute cost 24 (the cheap
        # third pair); at 16 MiB the memory must be 12 with higher recompute cost, showing FLOP
        # cost, not just fit, drives the choice.
        def f(xs, ws):
            # [FUNCTION DOC] HELPER: For each input and weight pair, matmul then cosine; returns
            # the sum.
            xs = [torch.mm(x, w).cos() for x, w in zip(xs, ws)]
            return sum(x.sum() for x in xs)

        data = [(4, 4), (6, 2), (2, 6)]
        xs, ws = zip(*[create_pair(a, b) for a, b in data])

        def call():
            # [FUNCTION DOC] HELPER: Calls f with the enclosing tuples of inputs and weights.
            return f(xs, ws)

        eager_mem, eager_flops = get_mem_and_flops(call)
        self.assertEqual(eager_mem, 40)
        self.assertEqual(eager_flops, 320)
        mem, flops = get_mem_and_flops(call, memory_budget=28 / eager_mem)
        # Save w1 and w2
        self.assertEqual(mem, 28)
        # We're recomputing w3 (the cheap one!)
        self.assertEqual(flops - eager_flops, 2 * 2 * 6)
        mem, flops = get_mem_and_flops(call, memory_budget=16 / eager_mem)
        # Save w2. Note that even though saving w1 gets us closer to our memory
        # limit, w2 is actually *more* FLOPs than w1!
        self.assertEqual(mem, 12)
        self.assertEqual(flops - eager_flops, 2 * 2 * 6 + 4 * 4 * 4)

    def test_attention_vs_linear(self):
        # [FUNCTION DOC] Compares recompute choice between a causal-free attention and a linear
        # layer for several sequence length and width pairs. Each pair states whether the matmul
        # or the attention should be recomputed (attention when the sequence is long relative to
        # the width). Defines helpers f and try_seq_length and calls the latter seven times.
        def f(x, w):
            # [FUNCTION DOC] HELPER: Treats x as a single head of attention input, runs scaled
            # dot product attention with itself as query, key and value, restores the shape,
            # applies a matmul with w and cosine, and returns the sum.
            orig_shape = x.shape
            x = x.reshape(1, 1, x.shape[0], x.shape[1])
            # I know this isn't technically right lol
            x = torch.nn.functional.scaled_dot_product_attention(
                x, x, x, is_causal=False
            ).reshape(*orig_shape)
            x = torch.mm(x, w)
            x = x.cos()
            return x.sum()

        def try_seq_length(S, D, expected_recompute):
            # [FUNCTION DOC] HELPER: For a sequence factor S and width factor D, counts matmul
            # and attention FLOPs of f in normalized units; checks eager memory is S*D*2, that a
            # 0.6 budget gives S*D memory, and that the extra FLOPs equal those of either
            # attention or matmul according to expected_recompute.
            x = torch.randn(S * 512, D * 512, requires_grad=True)
            w = torch.randn(D * 512, D * 512, requires_grad=True)

            def call():
                # [FUNCTION DOC] HELPER: Calls f with the enclosing x and w.
                return f(x, w)

            with FlopCounterMode(display=False) as mode:
                call()
            mm_flops = mode.get_flop_counts()["Global"][torch.ops.aten.mm]
            attn_flops = mode.get_total_flops() - mm_flops
            mm_flops /= 512**3 * 2
            attn_flops /= 512**3 * 2

            eager_mem, eager_flops = get_mem_and_flops(call)
            self.assertEqual(eager_mem, S * D * 2)

            mem, flops = get_mem_and_flops(
                call, memory_budget=0.6
            )  # Force it to recompute one of mm or attn
            self.assertEqual(mem, S * D)
            if expected_recompute == "attn":
                expected_flops = attn_flops
            else:
                expected_flops = mm_flops
            self.assertEqual(flops - eager_flops, expected_flops)

        # General behind this test is that if sequence length * 2 > D, then
        # attention is more expensive than the linear.
        try_seq_length(1, 1, "mm")
        try_seq_length(1, 3, "attn")
        try_seq_length(2, 2, "mm")
        try_seq_length(2, 1, "mm")
        try_seq_length(2, 5, "attn")
        try_seq_length(4, 7, "mm")
        try_seq_length(4, 9, "attn")

    def test_manual_ac(self):
        # [FUNCTION DOC] Checks that user checkpoint regions are respected under the auto memory
        # budget: a function with chained matmuls is checkpointed twice, and with a very small
        # budget memory and FLOPs must match the eager run (no extra recompute).
        # test that manual checkpoint boundaries are respected
        # when autoac is set
        def f(x):
            # [FUNCTION DOC] HELPER: Four chained matmuls (including x with its transpose);
            # returns the result.
            tmp1 = torch.matmul(x, x.T)
            tmp1 = torch.matmul(tmp1, tmp1)
            tmp1 = torch.matmul(tmp1, tmp1)
            out = torch.matmul(tmp1, x)
            return out

        def g(x):
            # [FUNCTION DOC] HELPER: Applies non-reentrant checkpointing of f twice in sequence.
            x = checkpoint(f, x, use_reentrant=False)
            x = checkpoint(f, x, use_reentrant=False)
            return x

        x = torch.randn(64, 1024, requires_grad=True)

        def call():
            # [FUNCTION DOC] HELPER: Sums the output of g on the enclosing x.
            return g(x).sum()

        eager_mem, eager_flops = get_mem_and_flops(call)
        # give the memory budget logic a value that should cause it to run,
        # but not recompute the matmuls
        mem, flops = get_mem_and_flops(call, memory_budget=0.01)
        self.assertEqual(mem, eager_mem)
        self.assertEqual(flops, eager_flops)


if __name__ == "__main__":
    # I'm using the cuda memory allocator to verify memory allocations
    if HAS_CUDA_AND_TRITON and not TEST_WITH_ROCM:
        run_tests()
