# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-GPU tests for TokenSwitchNCCL (torch.distributed._token_switch), the
# NCCL-EP based token dispatch/combine helper used for expert-parallel routing. Every test
# routes all tokens of rank r to rank (r + 1) modulo world size and checks the received
# values, the dispatch/combine roundtrip, and autograd gradients. The whole test class is
# skipped unless the torch._nccl_ep extension can really be imported. Defines two
# MultiProcContinuousTest classes (backend "nccl" and "nccl2"). No import-time side effects
# beyond a module logger.


import logging

import torch
import torch.distributed as dist
from torch.distributed._token_switch import _import_nccl_ep, TokenSwitchNCCL
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
)


log = logging.getLogger(__name__)


def _nccl_ep_available() -> bool:
    # [FUNCTION DOC] HELPER: Returns True only if CUDA is available and importing the
    # torch._nccl_ep extension via _import_nccl_ep succeeds. Any exception from the import
    # is swallowed (logged at debug level) and yields False. No other side effects.
    # The torch._nccl_ep extension is built only with USE_NCCL_EP, and a
    # USE_SYSTEM_NCCL=ON build additionally needs the nccl4py wheel at runtime.
    # Actually importing it is the real check: find_spec would only locate the
    # extension without dlopening it, so it would miss a missing libnccl_ep.so.
    if not torch.cuda.is_available():
        return False
    try:
        _import_nccl_ep()
    except Exception:
        log.debug("torch._nccl_ep unavailable; skipping EP tests", exc_info=True)
        return False
    return True


def requires_nccl_ep():
    # [FUNCTION DOC] HELPER: Returns a skip decorator (skip_but_pass_in_sandcastle_if) that
    # skips the test, with an explanatory message, when NCCL-EP is unavailable. The
    # availability check runs once at decoration time.
    return skip_but_pass_in_sandcastle_if(
        not _nccl_ep_available(),
        "Test requires a USE_NCCL_EP build (plus nccl4py for USE_SYSTEM_NCCL=ON)",
    )


NUM_TOKENS = 16
TOP_K = 1
HIDDEN = 64
TOKEN_SIZE_BYTES = HIDDEN * 2
NUM_MULTI_ROUND_DISPATCH_COMBINE = 3


def _generate_topk(rank, world_size, num_tokens, top_k, device):
    # [FUNCTION DOC] HELPER: Builds deterministic routing inputs for this rank. Returns
    # (topk_idx, topk_weights): topk_idx is an int64 tensor of shape (num_tokens, top_k)
    # filled with the next rank (rank + 1, wrapping around) as the expert id; topk_weights
    # is a float32 tensor of the same shape filled with 1 / top_k. Both live on device.
    remote_expert = (rank + 1) % world_size
    topk_idx = torch.full(
        (num_tokens, top_k), remote_expert, dtype=torch.int64, device=device
    )
    topk_weights = torch.full(
        (num_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device=device
    )
    return topk_idx, topk_weights


@requires_nccl_ep()
class TokenSwitchNCCLTest(MultiProcContinuousTest):
    # [CLASS DOC] Multi-process test suite (one process per GPU, nccl backend) for
    # TokenSwitchNCCL. State: a class-level cached TokenSwitchNCCL instance shared by all
    # tests of the class so it is only created once per process. Each test expects every
    # rank to send all of its tokens to the next rank. Needs at least 2 GPUs.
    _cached_token_switch: TokenSwitchNCCL | None = None

    @classmethod
    def backend_str(cls):
        # [FUNCTION DOC] HELPER: Returns the process group backend name, "nccl".
        return "nccl"

    @property
    def device(self):
        # [FUNCTION DOC] HELPER: Returns the CUDA device whose index equals this process rank.
        return torch.device("cuda", self.rank)

    @classmethod
    def get_token_switch(cls) -> TokenSwitchNCCL:
        # [FUNCTION DOC] HELPER: Lazily creates and caches the TokenSwitchNCCL for the default
        # process group, sized for NUM_TOKENS send tokens, world_size * NUM_TOKENS receive
        # tokens and TOKEN_SIZE_BYTES per token. Prints a message and does a barrier before
        # creation (collective, so all ranks must call it). Later calls return the cached one.
        if cls._cached_token_switch is None:
            pg = dist.distributed_c10d._get_default_group()
            rank = dist.get_rank(pg)
            world_size = dist.get_world_size(pg)
            print(f"rank {rank} creating token switch")
            dist.barrier(pg)
            cls._cached_token_switch = TokenSwitchNCCL(
                pg, world_size, NUM_TOKENS, world_size * NUM_TOKENS, TOKEN_SIZE_BYTES
            )
        return cls._cached_token_switch

    def _init(self):
        # [FUNCTION DOC] HELPER: Sets the current CUDA device to this rank's device and waits
        # on a barrier so all ranks start the test together.
        torch.cuda.set_device(self.device)
        dist.barrier()

    @skip_if_lt_x_gpu(2)
    def test_create_routing(self):
        # [FUNCTION DOC] Calls create_routing with every token routed to the next rank and
        # checks that the int32 per_expert_counts output (one local expert) is filled in
        # place with NUM_TOKENS, i.e. all tokens from the previous rank land on this rank.
        self._init()
        ts = self.get_token_switch()
        num_experts = self.world_size
        num_local_experts = num_experts // self.world_size
        topk_idx, _topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            num_local_experts, dtype=torch.int32, device=self.device
        )
        ts.create_routing(topk_idx, per_expert_counts)
        self.assertEqual(per_expert_counts.dtype, torch.int32)
        self.assertEqual(per_expert_counts.numel(), num_local_experts)
        self.assertEqual(per_expert_counts.item(), NUM_TOKENS)
        torch.cuda.synchronize()

    @skip_if_lt_x_gpu(2)
    def test_dispatch(self):
        # [FUNCTION DOC] Dispatches constant-valued bfloat16 tokens (value rank + 1) into
        # preallocated output buffers (tokens, weights, indices) and asserts that the first
        # NUM_TOKENS received rows all equal the value sent by the previous rank.
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            self.world_size, dtype=torch.int32, device=self.device
        )
        routing = ts.create_routing(topk_idx, per_expert_counts)

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )

        ts.dispatch(
            routing,
            tokens,
            topk_weights,
            out=(out_tokens, out_topk_weights, out_topk_idx),
        )
        torch.cuda.synchronize()

        src_rank = (self.rank - 1) % self.world_size
        expected_val = float(src_rank + 1)
        received = out_tokens[:NUM_TOKENS].float()
        self.assertTrue(
            received.eq(expected_val).all(),
            lambda msg: f"{msg}\nrank {self.rank}: expected {expected_val}, got {received[0, 0].item()}",
        )

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_roundtrip(self):
        # [FUNCTION DOC] Dispatches tokens to the next rank, feeds the received rows unchanged
        # into combine (output buffer provided), and asserts the combined result equals the
        # originally sent tokens (identity expert roundtrip).
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            self.world_size, dtype=torch.int32, device=self.device
        )
        routing = ts.create_routing(topk_idx, per_expert_counts)

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )

        ts.dispatch(
            routing,
            tokens,
            topk_weights,
            out=(out_tokens, out_topk_weights, out_topk_idx),
        )
        torch.cuda.synchronize()

        expert_tokens = out_tokens[:NUM_TOKENS].contiguous()
        combined = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        ts.combine(routing, expert_tokens, out=combined)
        torch.cuda.synchronize()

        expected = torch.full((NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16)
        self.assertEqual(combined.cpu(), expected)

    @skip_if_lt_x_gpu(2)
    def test_dispatch_autograd_backward(self):
        # [FUNCTION DOC] Runs dispatch (returning new output tensors) on tokens that require
        # grad, backpropagates the sum of the dispatched tokens, and asserts the input tokens
        # receive an all-ones gradient (backward routes gradients back via combine).
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            self.world_size, dtype=torch.int32, device=self.device
        )
        routing = ts.create_routing(topk_idx, per_expert_counts)

        tokens = torch.full(
            (NUM_TOKENS, HIDDEN),
            float(self.rank + 1),
            dtype=torch.bfloat16,
            device=self.device,
            requires_grad=True,
        )
        out_tokens, _out_weights, _out_idx = ts.dispatch(
            routing, tokens, topk_weights, num_recv_tokens
        )
        out_tokens.sum().backward()
        torch.cuda.synchronize()

        # grad_out_tokens is all-ones; combine routes them back: each token gets 1.0 per top-k slot
        self.assertIsNotNone(tokens.grad)
        self.assertEqual(tokens.grad, torch.ones_like(tokens))

    @skip_if_lt_x_gpu(2)
    def test_combine_autograd_backward(self):
        # [FUNCTION DOC] Primes the routing with a non-autograd dispatch, then runs combine on
        # expert tokens that require grad, backpropagates the sum, and asserts the expert
        # tokens get an all-ones gradient (backward of combine acts as a dispatch).
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            self.world_size, dtype=torch.int32, device=self.device
        )
        routing = ts.create_routing(topk_idx, per_expert_counts)

        # Pre-dispatch so the routing handle is primed, then test combine autograd
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN),
            float(self.rank + 1),
            dtype=torch.bfloat16,
            device=self.device,
        )
        out_tokens_buf = torch.zeros(
            num_recv_tokens, HIDDEN, dtype=torch.bfloat16, device=self.device
        )
        out_weights_buf = torch.zeros(
            num_recv_tokens, TOP_K, dtype=torch.float32, device=self.device
        )
        out_idx_buf = torch.zeros(
            num_recv_tokens, TOP_K, dtype=torch.int64, device=self.device
        )
        ts.dispatch(
            routing,
            tokens,
            topk_weights,
            out=(out_tokens_buf, out_weights_buf, out_idx_buf),
        )
        torch.cuda.synchronize()

        expert_tokens = (
            out_tokens_buf[:NUM_TOKENS].contiguous().detach().requires_grad_(True)
        )
        combined = ts.combine(routing, expert_tokens)
        combined.sum().backward()
        torch.cuda.synchronize()

        # grad_out_tokens is all-ones; dispatch routes them back to expert ranks
        self.assertIsNotNone(expert_tokens.grad)
        self.assertEqual(expert_tokens.grad, torch.ones_like(expert_tokens))

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_autograd_roundtrip(self):
        # [FUNCTION DOC] Chains autograd-enabled dispatch, an identity expert (slice of the
        # received rows) and combine, backpropagates the sum, and asserts the original
        # tokens get an all-ones gradient.
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            self.world_size, dtype=torch.int32, device=self.device
        )
        routing = ts.create_routing(topk_idx, per_expert_counts)

        token_val = float(self.rank + 1)
        tokens = torch.full(
            (NUM_TOKENS, HIDDEN),
            token_val,
            dtype=torch.bfloat16,
            device=self.device,
            requires_grad=True,
        )
        dispatched, _weights, _idx = ts.dispatch(
            routing, tokens, topk_weights, num_recv_tokens
        )
        expert_out = dispatched[:NUM_TOKENS].contiguous()
        combined = ts.combine(routing, expert_out)
        combined.sum().backward()
        torch.cuda.synchronize()

        # With identity expert (no-op), gradient should round-trip back as all-ones
        self.assertIsNotNone(tokens.grad)
        self.assertEqual(tokens.grad, torch.ones_like(tokens))

    @skip_if_lt_x_gpu(2)
    def test_dispatch_combine_multiple_rounds(self):
        # [FUNCTION DOC] Reuses one routing and the same preallocated buffers for
        # NUM_MULTI_ROUND_DISPATCH_COMBINE dispatch+combine rounds with a different token
        # value each round, and asserts the final combined output equals the last round's value.
        self._init()
        ts = self.get_token_switch()
        num_recv_tokens = self.world_size * NUM_TOKENS

        topk_idx, topk_weights = _generate_topk(
            self.rank, self.world_size, NUM_TOKENS, TOP_K, self.device
        )
        per_expert_counts = torch.zeros(
            self.world_size, dtype=torch.int32, device=self.device
        )
        routing = ts.create_routing(topk_idx, per_expert_counts)

        out_tokens = torch.zeros(
            (num_recv_tokens, HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        out_topk_weights = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.float32, device=self.device
        )
        out_topk_idx = torch.zeros(
            (num_recv_tokens, TOP_K), dtype=torch.int64, device=self.device
        )
        combined = torch.zeros(
            (NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device=self.device
        )

        for r in range(NUM_MULTI_ROUND_DISPATCH_COMBINE):
            token_val = float(self.rank + 1 + r)
            tokens = torch.full(
                (NUM_TOKENS, HIDDEN),
                token_val,
                dtype=torch.bfloat16,
                device=self.device,
            )
            ts.dispatch(
                routing,
                tokens,
                topk_weights,
                out=(out_tokens, out_topk_weights, out_topk_idx),
            )
            torch.cuda.synchronize()

            expert_tokens = out_tokens[:NUM_TOKENS].contiguous()
            ts.combine(routing, expert_tokens, out=combined)
            torch.cuda.synchronize()

        expected = torch.full(
            (NUM_TOKENS, HIDDEN), token_val, dtype=torch.bfloat16, device=self.device
        )
        self.assertEqual(combined, expected)


class TokenSwitchNCCL2Test(TokenSwitchNCCLTest):
    # [CLASS DOC] Re-runs all TokenSwitchNCCLTest tests with the "nccl2" backend. Has its own
    # class-level token switch cache so the instance is not shared with the parent class.
    _cached_token_switch: TokenSwitchNCCL | None = None

    @classmethod
    def backend_str(cls):
        # [FUNCTION DOC] HELPER: Returns the process group backend name, "nccl2".
        return "nccl2"


if __name__ == "__main__":
    run_tests()
