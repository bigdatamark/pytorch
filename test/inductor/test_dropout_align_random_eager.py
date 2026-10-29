# Owner(s): ["module: inductor"]
# [MODULE DOC] CUDA tests that Inductor-compiled dropout matches eager dropout when the
# align_random_eager config is on: same seed gives (nearly) identical masks and the same RNG
# offset advance. Contains module constants, two small model classes, seeding and timing
# helpers, a standalone parity helper and the test class. Several tests are skipped for CI
# failures, three primitive random tests are expected failures, and the main block runs only on
# Linux with CUDA and Triton.

import struct
import time
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
    requires_gpu,
)


# ───────────────────────────────────────────────────────────────
# Global config
# ───────────────────────────────────────────────────────────────
BASE_SEED = 1234
DROPOUT_P = 0.5
FFN_DIM = 3072
HIDDEN_DIM = 1024
BATCH = 3
SEQ_LEN = 512


# ───────────────────────────────────────────────────────────────
# Model under test
# ───────────────────────────────────────────────────────────────
class LinearBlock(torch.nn.Module):
    # [CLASS DOC] Feed-forward block: linear to the ffn size, dropout, relu, linear back to the
    # hidden size, held in a sequential container.
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = DROPOUT_P):
        # [FUNCTION DOC] Builds the sequential stack with the given dimensions and dropout
        # probability.
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, ffn_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor):
        # [FUNCTION DOC] Applies the sequential stack to the input and returns the result.
        return self.net(x)


class MultiDropoutBlock(torch.nn.Module):
    # [CLASS DOC] Like the feed-forward block but with three dropout layers (after the first
    # linear, after relu, and at the end) to stress random-number alignment across several
    # dropouts.
    """Block with multiple Dropout ops to stress RNG alignment."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = DROPOUT_P):
        # [FUNCTION DOC] Builds the stack: linear, dropout, relu, dropout, linear, dropout.
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, ffn_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, hidden_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        # [FUNCTION DOC] Applies the sequential stack to the input and returns the result.
        return self.net(x)


def build_models(dropout: float, *, mode=None, dynamic: bool = False):
    # [FUNCTION DOC] Builds an eager feed-forward block and a second one with the same weights
    # copied from the first, compiles the second with the given mode and dynamic flag, and
    # returns both as a pair.
    eager = LinearBlock(HIDDEN_DIM, FFN_DIM, dropout)
    compiled = LinearBlock(HIDDEN_DIM, FFN_DIM, dropout)
    compiled.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled, mode=mode, dynamic=dynamic)
    return eager, compiled


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def _set_seed(base: int = BASE_SEED):
    # [FUNCTION DOC] HELPER: Seeds the CPU generator and, if CUDA is available, the CUDA
    # generator with the given base seed (default is the module constant).
    torch.manual_seed(base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base)


def _sync(x: torch.Tensor):
    # [FUNCTION DOC] HELPER: Calls CUDA synchronize if the tensor lives on a CUDA device;
    # otherwise does nothing.
    if x.is_cuda:
        torch.cuda.synchronize()


def _timed_run(model, x, backward: bool = False):
    # [FUNCTION DOC] HELPER: Runs the model on x (optionally followed by a backward pass on the
    # mean of the squared output) between synchronizations and returns the elapsed wall time in
    # milliseconds and the output.
    _sync(x)
    t0 = time.time()
    y = model(x)
    if backward:
        (y.square().mean()).backward()
    _sync(x)
    return (time.time() - t0) * 1e3, y


def _cuda_rng_u64_seed_off():
    # [FUNCTION DOC] HELPER: Reads the CUDA RNG state bytes and returns the seed (first 8 bytes)
    # and the offset (bytes 8 to 24, which holds 8 bytes) decoded as little-endian unsigned
    # 64-bit integers.
    """Return (seed, offset) extracted from torch.cuda.get_rng_state()."""
    st = torch.cuda.get_rng_state()
    seed = struct.unpack("<Q", st[0:8].cpu().numpy().tobytes())[0]
    off = struct.unpack("<Q", st[8:24].cpu().numpy().tobytes())[0]
    return seed, off


def set_seed(seed):
    # [FUNCTION DOC] HELPER: Seeds both the CPU and the CUDA generators with the given seed.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def dropout_parity(shape, p=0.3, dtype=torch.float32, seed=1234):
    # [FUNCTION DOC] HELPER: Resets Dynamo, then runs an eager and a compiled training-mode
    # dropout on a tensor of ones on the CUDA device, re-seeding before each. Returns whether
    # the nonzero masks are identical, plus both outputs. Requires CUDA.
    """Returns (masks_equal, eager_out, compiled_out)."""
    DEVICE = torch.device("cuda")
    torch._dynamo.reset()
    x = torch.ones(shape, device=DEVICE, dtype=dtype)
    drop_e = torch.nn.Dropout(p).to(DEVICE).train()
    drop_c = torch.compile(torch.nn.Dropout(p).to(DEVICE).train())

    set_seed(seed)
    out_e = drop_e(x)
    set_seed(seed)
    out_c = drop_c(x)

    masks_eq = torch.equal(out_e != 0, out_c != 0)
    return masks_eq, out_e, out_c


# ───────────────────────────────────────────────────────────────
# Test class (Inductor idioms)
# ───────────────────────────────────────────────────────────────
@unittest.skipIf(
    not (IS_LINUX and HAS_CUDA_AND_TRITON),
    "Inductor CUDA dropout alignment tests require Linux and CUDA",
)
@config.patch(align_random_eager=True)
class TestDropoutAlignRandomEager(InductorTestCase):
    # [CLASS DOC] CUDA dropout alignment tests, skipped unless on Linux with CUDA and Triton and
    # run with the align_random_eager config patched on. Provides a tolerant mismatch-fraction
    # assertion.
    def assertSmallMismatchFraction(self, a, b, atol=1e-5, max_fraction=1e-3):
        # [FUNCTION DOC] Computes the absolute difference of the two tensors, counts elements
        # exceeding atol, and fails the test if the fraction of such elements exceeds
        # max_fraction. Tolerates a rare mask bit flip from the Philox conversion that a linear
        # layer would amplify.
        """Assert that only a small fraction of elements differ significantly.

        The Philox uint32→float32 conversion can produce values on opposite
        sides of the dropout threshold for ~1 in 10⁶ elements (architecture-
        and seed-dependent).  One wrong mask bit is then amplified by the
        downstream linear layer, so we check the *fraction* of large
        mismatches rather than requiring every element to be close.
        """
        diff = (a - b).abs()
        bad = (diff > atol).sum().item()
        total = diff.numel()
        fraction = bad / total
        self.assertLessEqual(
            fraction,
            max_fraction,
            f"Mismatch fraction {fraction:.6f} ({bad}/{total}) exceeds {max_fraction}",
        )

    @requires_gpu()
    @unittest.skip(
        "Disabled due to CI failures; see "
        "https://github.com/pytorch/pytorch/issues/190237"
    )
    def test_linear_block_compile_parity_forward(self):
        # [FUNCTION DOC] For eval and training modes, builds eager and compiled blocks on the
        # GPU, seeds identically before each forward under no_grad, and checks outputs agree up
        # to the mismatch fraction. Currently skipped for CI failures (issue 190237).
        device = torch.device(GPU_TYPE)

        for training in (False, True):
            eager, compiled = build_models(DROPOUT_P)
            eager.to(device)
            compiled.to(device)

            if training:
                eager.train()
                compiled.train()
            else:
                eager.eval()
                compiled.eval()

            x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

            # same seed before both runs (align dropout masks)
            _set_seed(BASE_SEED)
            with torch.no_grad():
                y_eager = eager(x)

            _set_seed(BASE_SEED)
            with torch.no_grad():
                y_comp = compiled(x)

            self.assertSmallMismatchFraction(y_eager, y_comp)

    @requires_gpu()
    @unittest.skip(
        "Disabled due to CI failures; see "
        "https://github.com/pytorch/pytorch/issues/190237"
    )
    def test_linear_block_compile_parity_backward(self):
        # [FUNCTION DOC] Runs forward and backward (loss is the mean of the squared output) on
        # eager and compiled blocks with identical seeds, then checks outputs and every
        # parameter gradient agree up to the mismatch fraction and that gradients exist.
        # Currently skipped for CI failures (issue 190237).
        device = torch.device(GPU_TYPE)

        eager, compiled = build_models(DROPOUT_P)
        eager.to(device)
        compiled.to(device)
        eager.train()
        compiled.train()

        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

        # eager fwd+bwd
        _set_seed(BASE_SEED)
        y_eager = eager(x)
        (y_eager.square().mean()).backward()

        # compiled fwd+bwd (Re-seed)
        for p in compiled.parameters():
            p.grad = None
        _set_seed(BASE_SEED)
        y_comp = compiled(x)
        (y_comp.square().mean()).backward()

        # outputs
        self.assertSmallMismatchFraction(y_eager.detach(), y_comp.detach())
        # grads
        for p_ref, p_new in zip(eager.parameters(), compiled.parameters()):
            self.assertIsNotNone(p_ref.grad)
            self.assertIsNotNone(p_new.grad)
            self.assertSmallMismatchFraction(p_ref.grad, p_new.grad)

    @requires_gpu()
    def test_dropout_mask_parity_and_rng_offset_cuda(self):
        # [FUNCTION DOC] For float32, float16 and bfloat16 (bf16 skipped if unsupported),
        # applies eager and compiled dropout to ones from the same seed, and compares mask
        # mismatch ratio (must be at most 1e-4), that the initial seed equals the base seed, and
        # that both advance the CUDA RNG offset by the same amount.
        device = torch.device(GPU_TYPE)
        H, W = BATCH * SEQ_LEN, FFN_DIM

        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        for dtype in dtypes:
            if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue

            x = torch.ones((H, W), device=device, dtype=dtype)

            # Eager
            _set_seed(BASE_SEED)
            seed0_e, off0_e = _cuda_rng_u64_seed_off()
            drop_e = torch.nn.Dropout(DROPOUT_P).to(device=device, dtype=dtype).train()
            mask_e = drop_e(x) != 0
            seed1_e, off1_e = _cuda_rng_u64_seed_off()
            delta_e = off1_e - off0_e

            # Compiled
            _set_seed(BASE_SEED)
            seed0_c, off0_c = _cuda_rng_u64_seed_off()
            drop_c = torch.nn.Dropout(DROPOUT_P).to(device=device, dtype=dtype)
            drop_c = torch.compile(drop_c)
            drop_c.train()
            mask_c = drop_c(x) != 0
            seed1_c, off1_c = _cuda_rng_u64_seed_off()
            delta_c = off1_c - off0_c

            mismatch_ratio = (mask_e != mask_c).float().mean().item()
            self.assertLessEqual(
                mismatch_ratio,
                1e-4,
                msg=lambda msg: f"{msg}\nDropout mask mismatch ratio too high: {mismatch_ratio:.8f}",
            )
            self.assertEqual(seed0_e, BASE_SEED)
            self.assertEqual(seed0_c, BASE_SEED)
            self.assertEqual(
                delta_e,
                delta_c,
                msg=lambda msg: f"{msg}\nRNG offset delta mismatch: eager={delta_e}, compiled={delta_c}",
            )

    # ───────────────────────────────────────────────────────────
    # multiple dropouts + multiple iterations
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    def test_multi_dropout_multi_iterations_parity(self):
        # [FUNCTION DOC] Over 10 iterations with a different seed each, runs eager and compiled
        # three-dropout blocks in training mode on random input with identical seeding and
        # requires the fraction of differing zero patterns of the outputs to be at most 1e-5.
        device = torch.device(GPU_TYPE)

        eager = MultiDropoutBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled = MultiDropoutBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled.load_state_dict(eager.state_dict())
        compiled = torch.compile(compiled)

        eager.train()
        compiled.train()

        num_iters = 10
        for i in range(num_iters):
            seed = BASE_SEED + i
            x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

            _set_seed(seed)
            y_eager = eager(x)

            _set_seed(seed)
            y_comp = compiled(x)

            mismatch_ratio = ((y_eager != 0) != (y_comp != 0)).float().mean().item()
            self.assertLessEqual(mismatch_ratio, 1e-5)

    # ───────────────────────────────────────────────────────────
    # dynamic shapes test (a)
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    @unittest.skip(
        "Disabled due to CI failures; see "
        "https://github.com/pytorch/pytorch/issues/190237"
    )
    def test_dropout_parity_dynamic_shapes(self):
        # [FUNCTION DOC] Compiles the block with dynamic shapes and compares to eager for two
        # sequence lengths with identical seeding, using the mismatch fraction check. Currently
        # skipped for CI failures (issue 190237).
        device = torch.device(GPU_TYPE)

        eager = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled.load_state_dict(eager.state_dict())
        compiled = torch.compile(compiled, dynamic=True)

        eager.train()
        compiled.train()

        shapes = [
            (BATCH, 512, HIDDEN_DIM),
            (BATCH, 128, HIDDEN_DIM),
        ]

        for shape in shapes:
            x = torch.randn(*shape, device=device)

            _set_seed(BASE_SEED)
            y_eager = eager(x)

            _set_seed(BASE_SEED)
            y_comp = compiled(x)

            self.assertSmallMismatchFraction(y_eager, y_comp)

    # ───────────────────────────────────────────────────────────
    # cudagraphs test via mode='reduce-overhead' (b)
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    @unittest.skip(
        "Disabled due to CI failures; see "
        "https://github.com/pytorch/pytorch/issues/190237"
    )
    def test_dropout_parity_cudagraphs_reduce_overhead(self):
        # [FUNCTION DOC] Compiles the block in reduce-overhead mode (CUDA graphs) and compares
        # one forward with eager under identical seeding. Currently skipped for CI failures
        # (issue 190237).
        device = torch.device(GPU_TYPE)

        eager = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled.load_state_dict(eager.state_dict())
        compiled = torch.compile(compiled, mode="reduce-overhead")

        eager.train()
        compiled.train()

        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

        _set_seed(BASE_SEED)
        y_eager = eager(x)

        _set_seed(BASE_SEED)
        y_comp = compiled(x)

        self.assertSmallMismatchFraction(y_eager, y_comp)

    # ───────────────────────────────────────────────────────────
    # Codegen sanity: run_and_get_code + FileCheck
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    def test_inductor_generated_code_contains_dropout(self):
        # [FUNCTION DOC] Compiles a training-mode block, captures the generated code of one
        # call, asserts at least one code string was produced, and uses FileCheck to require the
        # word dropout in the first.
        device = torch.device(GPU_TYPE)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

        model = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        model.train()
        compiled = torch.compile(model)

        def fn(inp):
            # [FUNCTION DOC] HELPER: Calls the compiled model on the input; passed to the code
            # capture utility.
            return compiled(inp)

        _, codes = run_and_get_code(fn, x)
        self.assertTrue(codes, msg="Expected inductor to generate at least one kernel")

        # Minimal sanity check that generated code mentions dropout.
        FileCheck().check("dropout").run(codes[0])

    # ───────────────────────────────────────────────────────────
    # Optional: perf smoke (GPU only)
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    def test_perf_smoke_cuda(self):
        # [FUNCTION DOC] Smoke test: times warmed-up eager and compiled evaluation-mode forward
        # passes and asserts both durations are positive.
        device = torch.device(GPU_TYPE)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

        eager, compiled = build_models(DROPOUT_P)
        eager.to(device)
        compiled.to(device)
        eager.eval()
        compiled.eval()

        # warm up
        _timed_run(eager, x, backward=False)
        _timed_run(compiled, x, backward=False)

        t_eager, _ = _timed_run(eager, x, backward=False)
        t_comp, _ = _timed_run(compiled, x, backward=False)

        self.assertGreater(t_comp, 0)
        self.assertGreater(t_eager, 0)

    # ───────────────────────────────────────────────────────────
    # Helper for primitive random parity (rand / randn / randint)
    # ───────────────────────────────────────────────────────────
    def _run_primitive_random_parity(self, kind, device, shape):
        # [FUNCTION DOC] HELPER: Compares eager and compiled random primitives. For kind rand,
        # randn or randint it defines an eager generator function on the device and shape and
        # compiles it; raises AssertionError for any other kind. Seeds identically, runs both,
        # and asserts exact equality.
        if kind == "rand":

            def eager():
                # [FUNCTION DOC] HELPER: Returns a uniform random tensor of the given shape on
                # the device.
                return torch.rand(shape, device=device)

            compiled = torch.compile(eager)

        elif kind == "randn":

            def eager():
                # [FUNCTION DOC] HELPER: Returns a standard normal random tensor of the given
                # shape on the device.
                return torch.randn(shape, device=device)

            compiled = torch.compile(eager)

        elif kind == "randint":

            def eager():
                # [FUNCTION DOC] HELPER: Returns random integers in the range from 0 to the
                # int32 maximum minus one, of the given shape on the device.
                return torch.randint(0, 2**31 - 1, shape, device=device)

            compiled = torch.compile(eager)

        else:
            raise AssertionError(f"unknown primitive random kind: {kind}")

        _set_seed(BASE_SEED)
        out_eager = eager()

        _set_seed(BASE_SEED)
        out_comp = compiled()

        torch.testing.assert_close(out_eager, out_comp, rtol=0.0, atol=0.0)

    # ───────────────────────────────────────────────────────────
    # Primitive random fns: rand / randn / randint -> mark as XFAIL
    # ───────────────────────────────────────────────────────────
    @unittest.expectedFailure
    @requires_gpu()
    def test_primitive_rand_parity(self):
        # [FUNCTION DOC] Expected failure: checks compiled uniform random output equals eager
        # exactly.
        device = torch.device(GPU_TYPE)
        shape = (BATCH, SEQ_LEN, HIDDEN_DIM)
        self._run_primitive_random_parity("rand", device, shape)

    @unittest.expectedFailure
    @requires_gpu()
    def test_primitive_randn_parity(self):
        # [FUNCTION DOC] Expected failure: checks compiled normal random output equals eager
        # exactly.
        device = torch.device(GPU_TYPE)
        shape = (BATCH, SEQ_LEN, HIDDEN_DIM)
        self._run_primitive_random_parity("randn", device, shape)

    @unittest.expectedFailure
    @requires_gpu()
    def test_primitive_randint_parity(self):
        # [FUNCTION DOC] Expected failure: checks compiled random integer output equals eager
        # exactly.
        device = torch.device(GPU_TYPE)
        shape = (BATCH, SEQ_LEN, HIDDEN_DIM)
        self._run_primitive_random_parity("randint", device, shape)

    # ───────────────────────────────────────────────────────────
    # nn.Dropout as primitive RNG consumer (should PASS)
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    def test_primitive_nn_dropout_parity(self):
        # [FUNCTION DOC] Applies eager and compiled dropout modules (same state) to ones with
        # identical seeding and requires exact equality of outputs.
        device = torch.device(GPU_TYPE)
        shape = (BATCH, SEQ_LEN, HIDDEN_DIM)

        x = torch.ones(shape, device=device)

        drop_eager = torch.nn.Dropout(DROPOUT_P).to(device).train()
        drop_compiled = torch.nn.Dropout(DROPOUT_P).to(device).train()
        drop_compiled.load_state_dict(drop_eager.state_dict())
        drop_compiled = torch.compile(drop_compiled)

        _set_seed(BASE_SEED)
        out_eager = drop_eager(x)

        _set_seed(BASE_SEED)
        out_comp = drop_compiled(x)

        torch.testing.assert_close(out_eager, out_comp, rtol=0.0, atol=0.0)

    # ───────────────────────────────────────────────────────────
    # Large seed (>32-bit) packing truncation
    # Seed and base are packed into int64 as (seed << 32) | base.
    # Seeds > 2^32 overflow.
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    def test_large_seed(self):
        # [FUNCTION DOC] For two seeds above 32 bits, compares eager and compiled dropout masks
        # via the parity helper in a subtest and requires the masks equal. The comment notes the
        # seed is packed with a base into one 64-bit integer so large seeds can overflow.
        for seed in [2**33 + 1, 2**40 + 12345]:
            with self.subTest(seed=seed):
                masks_eq, _, _ = dropout_parity((1024,), seed=seed)
                self.assertTrue(
                    masks_eq, lambda msg: f"{msg}\nseed={seed}: mask mismatch"
                )


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA_AND_TRITON:
        run_tests(needs="filelock")
