# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for DTensor context parallelism (CP) in
# torch.distributed.tensor.experimental._attention. Covers ring attention for SDPA with
# several backends, rotate methods and load balancers, CP with flex attention (causal and
# document masks, with various load balancers), the flex_cp_allgather custom op, sequence
# sharding helpers, argument handling of _ContextParallel input functions, and a CPU-only
# test of the per-document head/tail load balancer layout. Import-time side effects: builds
# the list of supported SDPA backends from platform flags, compiles flex_attention and
# create_block_mask at module level (lazily run), and creates local-tensor variants of
# several test classes.
import contextlib
import itertools
import random
import unittest
import unittest.mock
from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.experimental._attention import (
    _CausalBehavior,
    _context_parallel_shard,
    _ContextParallel,
    _cp_options,
    _disable_context_parallel_dispatcher,
    _enable_context_parallel_dispatcher,
    _HeadTailLoadBalancer,
    _is_causal_behavior,
    _LoadBalancer,
    _PerDocumentHeadTailLoadBalancer,
    _PTRRLoadBalancer,
    _RotateMethod,
    context_parallel,
    context_parallel_unshard,
    set_rotate_method,
)
from torch.distributed.tensor.experimental._context_parallel._cp_custom_ops import (
    flex_cp_allgather,
)
from torch.distributed.tensor.experimental._context_parallel._sharding_rules import (
    register_cp_sharding_rules,
    unregister_cp_sharding_rules,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    AuxOutput,
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_distributed import (
    skip_if_lt_x_gpu,
    skip_if_rocm_ver_atleast_multiprocess,
)
from torch.testing._internal.common_utils import run_tests, skipIfRocm, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    map_local_tensor_for_rank,
    with_comms,
)


c10d_functional = torch.ops.c10d_functional
backends = []
if PLATFORM_SUPPORTS_FLASH_ATTENTION:
    backends.append(SDPBackend.FLASH_ATTENTION)
if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
    backends.append(SDPBackend.EFFICIENT_ATTENTION)
if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
    backends.append(SDPBackend.CUDNN_ATTENTION)

rotater_enum_to_str = {
    _RotateMethod.ALL_GATHER: "allgather",
    _RotateMethod.ALL_TO_ALL: "alltoall",
}  # mapping from _RotateMethod enum to string


class SDPAWrapper(torch.nn.Module):
    # [CLASS DOC] HELPER: Module wrapping scaled_dot_product_attention so context-parallel
    # plans can be applied to it. Holds the attention callable (optionally torch.compiled with
    # fullgraph and the aot_eager backend) and the SDPBackend to force when calling.
    def __init__(self, compiled: bool, backend: SDPBackend) -> None:
        # [FUNCTION DOC] HELPER: Stores the attention function in self.sdpa (compiled when
        # compiled is true, else the plain function) and the backend in self.backend.
        super().__init__()
        if compiled:
            self.sdpa = torch.compile(
                F.scaled_dot_product_attention,
                fullgraph=True,
                backend="aot_eager",
            )
        else:
            self.sdpa = F.scaled_dot_product_attention
        self.backend = backend

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Calls the stored attention function with all given arguments
        # inside an sdpa_kernel context restricted to the stored backend; returns its output.
        with sdpa_kernel(self.backend):
            return self.sdpa(*args, **kwargs)


class RingAttentionTest(DTensorTestBase):
    # [CLASS DOC] Multi-GPU tests of ring attention for SDPA under context parallelism:
    # compares CP outputs and gradients with single-device SDPA across parameter grids, and
    # unit-tests the causal behavior helper and short-sequence handling. World size equals
    # the visible CUDA device count and the process group is kept alive between tests.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the number of visible CUDA devices as the world size.
        return torch.cuda.device_count()

    @property
    def destroy_pg_upon_exit(self) -> bool:
        # [FUNCTION DOC] Returns False so the process group is not destroyed on exit.
        return False

    @skip_if_lt_x_gpu(2)
    @skipIfRocm  # Missing _c10d_functional_autograd::all_to_all_single
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Does not support flash nor efficient attention",
    )
    @with_comms
    def test_ring_attention_sdpa(self) -> None:
        # [FUNCTION DOC] Runs _test_ring_attention_sdpa through run_subtests over the grid of
        # is_causal, compiled, SDPA backend, load balancing, rotate method (all-to-all or
        # all-gather), forward-only and use_context. Skipped with under 2 GPUs, on ROCm, or
        # without fused attention support.
        self.run_subtests(
            {
                "is_causal": [True, False],
                "compiled": [True, False],
                "backend": backends,
                "load_balance": [True, False],
                "rotater": [_RotateMethod.ALL_TO_ALL, _RotateMethod.ALL_GATHER],
                "test_forward_only": [True, False],
                "use_context": [True, False],
            },
            self._test_ring_attention_sdpa,
        )

    def _ring_attention_sdpa(
        self,
        cp_q: torch.Tensor,
        cp_k: torch.Tensor,
        cp_v: torch.Tensor,
        *,
        fn_eval: Callable,
        mesh: DeviceMesh,
        seq_dim: int,
        is_causal: bool,
        compiled: bool,
        backend: SDPBackend,
        rotater: _RotateMethod,
        test_forward_only: bool,
        load_balance: bool,
        use_context: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Runs SDPA on the given q, k, v under context parallelism and
        # returns (output, dq, dk, dv) of the sharded run. If use_context is false it applies a
        # _ContextParallel plan (SDPA) to an SDPAWrapper, shards inputs with an optional head-
        # tail load balancer and enables the CP dispatcher; otherwise it sets the global
        # load-balance option and enters context_parallel over the buffers, using the plain
        # functional SDPA (optionally compiled). Marks inputs as requiring grad, evaluates via
        # fn_eval inside the backend context, and when not compiled with all-to-all rotation
        # asserts the all_to_all_single count equals world_size-1 (forward only) or
        # 3*world_size-2. Afterwards resets requires_grad, and disables the dispatcher or exits
        # the context. Mutates the input tensors (sharded in place when using the context).
        if not use_context:
            cp_plan = _ContextParallel(
                seq_dim=seq_dim,
                attention_type=_ContextParallel.AttentionType.SDPA,
            )
            attention = SDPAWrapper(compiled=compiled, backend=backend)
            attention = parallelize_module(attention, mesh, cp_plan)
            if load_balance:
                seq_len = cp_q.size(seq_dim)
                load_balancer = _HeadTailLoadBalancer(seq_len, mesh.size(), cp_q.device)
            else:
                load_balancer = None
            cp_q, cp_k, cp_v = _context_parallel_shard(
                mesh, (cp_q, cp_k, cp_v), (seq_dim,) * 3, load_balancer=load_balancer
            )
            _enable_context_parallel_dispatcher()
        else:
            # Theoretically, context_parallel() should not be used to shard
            # parameters because when require_grad is True, resize_ is not
            # allowed. But requires_grad of cp_q, cp_k, and cp_v are False
            # now. So we can just use context_parallel() to shard q, k, v.
            # In reality, context_parallel() should be used to shard the input.
            # In reality, context_parallel() should only be used to shard
            # the model inputs (batch).

            _cp_options.enable_load_balance = load_balance
            cp_context = context_parallel(
                mesh, buffers=(cp_q, cp_k, cp_v), buffer_seq_dims=(seq_dim,) * 3
            )
            cp_context.__enter__()

            # NOTE: This demonstrates that monkey patching is not fully reliable.
            # If we use SDPAWrapper directly, the monkey patching dispatch mode
            # does not function correctly. To ensure proper behavior,
            # F.scaled_dot_product_attention must be referenced within the
            # context_parallel() scope.
            attention = F.scaled_dot_product_attention
            if compiled:
                attention = torch.compile(
                    attention, fullgraph=True, backend="aot_eager"
                )

        for target in [cp_q, cp_k, cp_v]:
            target.requires_grad = True

        check_comm_counts = not compiled and rotater == _RotateMethod.ALL_TO_ALL
        comm_mode = CommDebugMode() if check_comm_counts else contextlib.nullcontext()
        with comm_mode:
            with sdpa_kernel(backend):
                cp_out = fn_eval(
                    attention,
                    cp_q,
                    cp_k,
                    cp_v,
                    is_causal=is_causal,
                )

            if check_comm_counts:
                expect_all2all_count = (
                    self.world_size - 1
                    if test_forward_only
                    else self.world_size * 3 - 2
                )
                self.assertDictEqual(
                    comm_mode.get_comm_counts(),
                    {c10d_functional.all_to_all_single: expect_all2all_count},
                )
        cp_dq, cp_dk, cp_dv = cp_q.grad, cp_k.grad, cp_v.grad
        for target in [cp_q, cp_k, cp_v]:
            target.requires_grad = False

        if not use_context:
            _disable_context_parallel_dispatcher()
        else:
            cp_context.__exit__(None, None, None)

        return cp_out, cp_dq, cp_dk, cp_dv

    def _test_ring_attention_sdpa(
        self,
        is_causal: bool,
        compiled: bool,
        backend: SDPBackend,
        load_balance: bool,
        rotater: _RotateMethod,
        test_forward_only: bool,
        use_context: bool,
    ) -> None:
        # [FUNCTION DOC] HELPER: One subtest of the ring attention grid. Returns immediately
        # for load_balance without is_causal, and for compiled=True. Otherwise sets the
        # rotate method, builds random q, k, v (bf16 for flash/cudnn backends, else float32)
        # broadcast from rank 0, computes reference output and gradients with plain SDPA, runs
        # _ring_attention_sdpa on clones, unshards the results, and checks closeness with
        # backend-dependent tolerances (outputs, and gradients unless forward only).
        # KNOWN ISSUE: every compiled=True subtest returns early, so compiled CP is untested.
        def fn_eval(fn, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls fn with the given arguments, under no_grad if
            # test_forward_only; otherwise runs it, backpropagates the sum of the output, and
            # returns the output.
            if test_forward_only:
                with torch.no_grad():
                    return fn(*args, **kwargs)
            else:
                out = fn(*args, **kwargs)
                out.sum().backward()
                return out

        if load_balance and not is_causal:
            return

        # Compilation with context_parallel doesn't work yet — both paths
        # (use_context=True monkey-patch and use_context=False parallelize_module)
        # fail during tracing because DTensor dispatch interferes with sdpa.
        # Previously CommDebugMode was active for all subtests, which caused
        # the frame to be silently skipped, masking this limitation.
        if compiled:
            return

        set_rotate_method(rotater_enum_to_str[rotater])
        self.assertEqual(_cp_options.rotate_method, rotater)
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))
        bs = 8
        seq_length = 1024
        seq_dim = 2
        dim = 32
        nheads = 8
        torch.manual_seed(10)
        dtype = (
            torch.bfloat16
            if backend == SDPBackend.FLASH_ATTENTION
            or backend == SDPBackend.CUDNN_ATTENTION
            else torch.float32
        )

        q, k, v = [
            torch.rand(
                (bs, nheads, seq_length * self.world_size, dim),
                device=self.device_type,
                dtype=dtype,
                requires_grad=True,
            )
            for _ in range(3)
        ]

        # Ensure all ranks have the same initialization data.
        with torch.no_grad():
            dist.broadcast(q, src=0)
            dist.broadcast(k, src=0)
            dist.broadcast(v, src=0)

        with sdpa_kernel(backend):
            out = fn_eval(F.scaled_dot_product_attention, q, k, v, is_causal=is_causal)

        cp_q, cp_k, cp_v = [target.detach().clone() for target in [q, k, v]]
        cp_out, cp_dq, cp_dk, cp_dv = self._ring_attention_sdpa(
            cp_q,
            cp_k,
            cp_v,
            fn_eval=fn_eval,
            mesh=device_mesh,
            seq_dim=seq_dim,
            is_causal=is_causal,
            compiled=compiled,
            backend=backend,
            rotater=rotater,
            test_forward_only=test_forward_only,
            load_balance=load_balance,
            use_context=use_context,
        )

        # Due to numerical error, we need to choose different atol for different
        # attention kernels
        (cp_out,) = context_parallel_unshard(device_mesh, [cp_out], [seq_dim])
        atol = (
            2e-06
            if backend == SDPBackend.EFFICIENT_ATTENTION
            else 8e-3 * self.world_size
        )
        rtol = (
            1e-05
            if backend == SDPBackend.EFFICIENT_ATTENTION
            else 1e-3 * self.world_size
        )
        torch.testing.assert_close(out, cp_out, atol=atol, rtol=rtol)

        if test_forward_only:
            return

        cp_dq, cp_dk, cp_dv = context_parallel_unshard(
            device_mesh,
            [cp_dq, cp_dk, cp_dv],
            [seq_dim] * 3,
        )
        torch.testing.assert_close(q.grad, cp_dq, atol=atol, rtol=rtol)
        torch.testing.assert_close(k.grad, cp_dk, atol=atol, rtol=rtol)
        torch.testing.assert_close(v.grad, cp_dv, atol=atol, rtol=rtol)

    def test_is_causal_behavior(self) -> None:
        # [FUNCTION DOC] Checks _is_causal_behavior: non-causal gives NOT_IS_CAUSAL; for
        # causal with world size 2 and load balancing off, rank 0 gives IS_CAUSAL then SKIP
        # and rank 1 gives IS_CAUSAL then NOT_IS_CAUSAL; with load balancing on, both ranks
        # give IS_CAUSAL then NOT_IS_CAUSAL. Mutates the global enable_load_balance option
        # and leaves it set to True.
        _cp_options.enable_load_balance = False
        self.assertEqual(
            _is_causal_behavior(rank=0, world_size=4, i=0, is_causal=False),
            _CausalBehavior.NOT_IS_CAUSAL,
        )

        ranks = [
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.SKIP],
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
        ]
        for rank, iters in enumerate(ranks):
            for i, behavior in enumerate(iters):
                self.assertEqual(
                    _is_causal_behavior(rank=rank, world_size=2, i=i, is_causal=True),
                    behavior,
                )

        _cp_options.enable_load_balance = True
        ranks = [
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
        ]
        for rank, iters in enumerate(ranks):
            for i, behavior in enumerate(iters):
                self.assertEqual(
                    _is_causal_behavior(rank=rank, world_size=2, i=i, is_causal=True),
                    behavior,
                )

    @skip_if_lt_x_gpu(2)
    @skipIfRocm
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    def test_context_parallel_sdpa_short_sequence(self) -> None:
        # [FUNCTION DOC] With load balancing requested and a sequence of length world_size,
        # for head dims 1, 2, 4, 8: compares flash-attention causal SDPA inside context_parallel
        # with the single-device result after unsharding. Asserts load balancing is turned off
        # inside the context, and restores the previous option value afterwards.
        old_load_balance = _cp_options.enable_load_balance
        try:
            _cp_options.enable_load_balance = True
            device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))
            qkv_len = self.world_size
            for dim in [1, 2, 4, 8]:
                with self.subTest(dim=dim):
                    qkv = [
                        torch.rand(
                            (1, 1, qkv_len, dim),
                            device=self.device_type,
                            dtype=torch.bfloat16,
                        )
                        for _ in range(3)
                    ]

                    with torch.no_grad():
                        for t in qkv:
                            dist.broadcast(t, src=0)

                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        out = F.scaled_dot_product_attention(*qkv, is_causal=True)

                    cp_qkv = [t.detach().clone() for t in qkv]
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        with context_parallel(
                            device_mesh, buffers=cp_qkv, buffer_seq_dims=[2, 2, 2]
                        ):
                            self.assertFalse(_cp_options.enable_load_balance)
                            cp_out = F.scaled_dot_product_attention(
                                *cp_qkv, is_causal=True
                            )

                        (cp_out,) = context_parallel_unshard(device_mesh, [cp_out], [2])

                    torch.testing.assert_close(
                        cp_out,
                        out,
                        atol=8e-3 * self.world_size,
                        rtol=1e-3 * self.world_size,
                    )
        finally:
            _cp_options.enable_load_balance = old_load_balance


# Compile the flex_attention function
compiled_flex_attention = torch.compile(flex_attention, dynamic=False, fullgraph=True)
compiled_create_block_mask = torch.compile(
    create_block_mask, dynamic=False, fullgraph=True
)


def causal_mask(b, h, q_idx, kv_idx):
    # [FUNCTION DOC] HELPER: Flex attention mask function; true where the query index is at
    # or after the key/value index (causal). Batch and head indices are ignored.
    return q_idx >= kv_idx


# copied from https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/masks/document_mask.py
def generate_random_lengths(total_length, num_documents) -> list[int]:
    # [FUNCTION DOC] HELPER: Returns num_documents positive lengths summing to total_length:
    # every document starts at 1 and each remaining token is given to a uniformly random
    # document using the global random module. Assumes total_length >= num_documents.
    # Initialize all lengths to 1 to ensure each document has at least one token
    lengths = [1] * num_documents
    remaining_length = total_length - num_documents

    # Randomly distribute the remaining length
    for _ in range(remaining_length):
        index = random.randint(0, num_documents - 1)
        lengths[index] += 1

    return lengths


def generate_random_lengths_in_chunks(
    total_length, num_documents, chunk_size
) -> list[int]:
    # [FUNCTION DOC] HELPER: Returns num_documents lengths, each a positive multiple of
    # chunk_size, summing to total_length, by randomly distributing chunks among documents
    # with the global random module. Raises AssertionError if total_length is not divisible
    # by chunk_size or there are fewer chunks than documents.
    # Generate a list of random document lengths so that each document contains
    # some number of chunks of size `chunk_size`. This means each document's length
    # must be a multiple of `chunk_size`. Besides, the lengths of all the documents
    # sum up to `total_length`.
    num_chunks = total_length // chunk_size
    if not (total_length % chunk_size == 0 and num_chunks >= num_documents):
        raise AssertionError(
            f"total_length % chunk_size == {total_length % chunk_size} (expected 0), "
            f"num_chunks={num_chunks} vs num_documents={num_documents}"
        )

    num_chunks_per_document = [1] * num_documents
    remaining_chunks = num_chunks - num_documents
    # Randomly distribute the remaining chunks
    for _ in range(remaining_chunks):
        index = random.randint(0, num_documents - 1)  # document_id
        num_chunks_per_document[index] += 1

    return [num_chunks * chunk_size for num_chunks in num_chunks_per_document]


def length_to_offsets(lengths: list[list[int]], device: str | torch.device) -> Tensor:
    # [FUNCTION DOC] HELPER: Converts per-batch lists of document lengths to an int32 tensor
    # of cumulative offsets on the given device, shape (batch, num_documents + 1), each row
    # starting at 0.
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets = [[0] + lengths_in_batch for lengths_in_batch in lengths]
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets


def _offsets_to_doc_ids_tensor(offsets):
    # [FUNCTION DOC] HELPER: For each batch row of the offsets tensor, repeats the document
    # index (int32) once per token in that document, and stacks the rows into a (batch,
    # seq_len) tensor of document ids. Requires all rows to cover the same total length.
    doc_ids = []
    device = offsets.device
    for batch_idx in range(offsets.size(0)):
        counts = offsets[batch_idx][1:] - offsets[batch_idx][:-1]
        doc_id = torch.repeat_interleave(
            torch.arange(len(counts), device=device, dtype=torch.int32), counts
        )
        doc_ids.append(doc_id)

    return torch.stack(doc_ids)


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature, offsets: Tensor
) -> _mask_mod_signature:
    # [FUNCTION DOC] HELPER: Builds a document-aware mask function from mask_mod and a batch
    # of offsets: precomputes per-token document ids and returns a closure applying mask_mod
    # to within-document positions combined with a same-document check.
    # KNOWN ISSUE: docstring says offsets has shape (num_documents + 1), but the code indexes
    # it per batch row (shape batch x (num_documents + 1)).
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """
    document_id = _offsets_to_doc_ids_tensor(offsets)

    def doc_mask_mod(b, h, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: Mask function true when query and key/value tokens are in the
        # same document and the wrapped mask_mod holds on their within-document positions.
        same_doc = document_id[b][q_idx] == document_id[b][kv_idx]
        q_logical = q_idx - offsets[b, document_id[b, q_idx]]
        kv_logical = kv_idx - offsets[b, document_id[b, kv_idx]]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask

    return doc_mask_mod


class FlexAttentionWrapper(torch.nn.Module):
    # [CLASS DOC] HELPER: Module wrapping a shared class-level torch.compile'd flex_attention
    # so a context-parallel plan can be applied to it. Has no parameters or state.
    _flex_attn: ClassVar[Callable] = torch.compile(flex_attention)

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Only calls the base module initializer.
        super().__init__()

    def forward(
        self, *args: object, **kwargs: object
    ) -> [
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, AuxOutput],
    ]:
        # [FUNCTION DOC] HELPER: Forwards all arguments to the compiled flex_attention and
        # returns its result (an output tensor, or output plus aux info when requested).
        # KNOWN ISSUE: the return annotation is a list literal rather than a valid type.
        return FlexAttentionWrapper._flex_attn(*args, **kwargs)


class CPFlexAttentionTest(DTensorTestBase):
    # [CLASS DOC] 2-GPU tests of context-parallel flex attention: for causal and document
    # masks with different load balancers, compares sharded outputs, log-sum-exp and q/k/v
    # gradients against single-device compiled flex attention.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the fixed world size of 2.
        return 2

    def _test_cp_flex_attention(
        self,
        *,
        qkv_size: int,
        B: int = 1,
        block_mask,
        lb_type: str,
        document_lengths: list[list[int]] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Compares context-parallel flex attention against the
        # single-device reference for the given block mask and load balancer name. Enables
        # deterministic algorithms globally and seeds CUDA. Builds float32 q, k, v (batch B if
        # B > 1 else 8, 8 heads, head dim 32), runs compiled flex attention with lse and
        # backward as reference, then shards q, k, v and the block mask along dim 2 over a 1-D
        # mesh named cp and runs a wrapper module with a FLEX _ContextParallel plan. Unshards
        # output, lse and gradients with the same balancer and checks closeness (atol 2e-6,
        # rtol 1e-5). May raise NotImplementedError from sharding for incompatible lengths.
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed(1234)

        dtype = torch.float32
        bs = B if B > 1 else 8
        dim = 32
        nheads = 8
        seq_dim = 2
        lb = self._get_load_balancer(
            lb_type,
            {
                "seq_length": qkv_size,
                "document_lengths": document_lengths,
                "block_mask": block_mask,
            },
        )

        qkv = [
            torch.rand(
                (bs, nheads, qkv_size, dim),
                device=self.device_type,
                dtype=dtype,
                requires_grad=True,
            )
            for _ in range(3)
        ]

        expect_out, expect_aux = compiled_flex_attention(
            *qkv, block_mask=block_mask, return_aux=AuxRequest(lse=True)
        )
        expect_out.sum().backward()

        # Prepare the required global vars for CP+Flex:
        device_mesh = init_device_mesh(
            device_type=self.device_type,
            mesh_shape=(self.world_size,),
            mesh_dim_names=("cp",),
        )

        flex_attention_wrapper_module = FlexAttentionWrapper()
        cp_plan = _ContextParallel(
            seq_dim=seq_dim,
            attention_type=_ContextParallel.AttentionType.FLEX,
        )
        parallelize_module(
            flex_attention_wrapper_module,
            device_mesh,
            cp_plan,
        )

        *cp_qkv, cp_block_mask = _context_parallel_shard(
            device_mesh,
            [t.detach().clone() for t in qkv] + [block_mask],
            [seq_dim] * 4,
            load_balancer=lb,
        )
        for t in cp_qkv:
            t.requires_grad = True

        cp_out, cp_aux = flex_attention_wrapper_module(
            *cp_qkv,
            block_mask=cp_block_mask,
            return_aux=AuxRequest(lse=True),
        )

        # backward run
        cp_out.sum().backward()

        atol = 2e-06
        rtol = 1e-05
        # unshard the output
        cp_out, cp_lse = context_parallel_unshard(
            device_mesh,
            buffers=[cp_out, cp_aux.lse],
            seq_dims=[seq_dim] * 2,
            load_balancer=lb,
        )
        torch.testing.assert_close(cp_out, expect_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(cp_lse, expect_aux.lse, atol=atol, rtol=rtol)

        # unshard the gradient
        cp_qkv_grad = context_parallel_unshard(
            device_mesh,
            buffers=[t.grad for t in cp_qkv],
            seq_dims=[seq_dim] * 3,
            load_balancer=lb,
        )

        qkv_grad = [t.grad for t in qkv]
        for grad, cp_grad in zip(qkv_grad, cp_qkv_grad):
            torch.testing.assert_close(grad, cp_grad, atol=atol, rtol=rtol)

    def _get_load_balancer(
        self, lb_type: str, kwargs: dict[str, Any]
    ) -> _LoadBalancer | None:
        # [FUNCTION DOC] HELPER: Builds the load balancer named by lb_type from the kwargs
        # mapping (seq_length, document_lengths, block_mask). "None" gives None; the head-tail
        # balancer needs an int seq_length, the per-document one needs a list of document
        # lengths, and the PTRR one needs a BlockMask. Raises AssertionError on a wrongly
        # typed input and ValueError for an unknown name.
        seq_length = kwargs["seq_length"]
        document_lengths = kwargs["document_lengths"]
        block_mask = kwargs["block_mask"]

        # generate load balancer
        if lb_type == "None":
            load_balancer = None  # no load-balance
        elif lb_type == "_HeadTailLoadBalancer":
            if not isinstance(seq_length, int):
                raise AssertionError(f"Expected int, got {type(seq_length)}")
            load_balancer = _HeadTailLoadBalancer(
                seq_length, self.world_size, torch.device(self.device_type)
            )
        elif lb_type == "_PerDocumentHeadTailLoadBalancer":
            if not isinstance(document_lengths, list):
                raise AssertionError(f"Expected list, got {type(document_lengths)}")
            load_balancer = _PerDocumentHeadTailLoadBalancer(
                document_lengths, self.world_size, torch.device(self.device_type)
            )
        elif lb_type == "_PTRRLoadBalancer":
            if not isinstance(block_mask, BlockMask):
                raise AssertionError(f"Expected BlockMask, got {type(block_mask)}")
            load_balancer = _PTRRLoadBalancer(
                block_mask,
                self.world_size,
            )
        else:
            raise ValueError(f"load_balancer type {lb_type} is not supported!")

        return load_balancer

    @skip_if_lt_x_gpu(2)
    @with_comms
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @skip_if_rocm_ver_atleast_multiprocess([7, 14])
    def test_cp_flex_attention_causal_mask(self) -> None:
        # [FUNCTION DOC] Runs the CP flex comparison with a causal block mask over sequence
        # lengths (256 * world_size, 2048) and load balancers (none, head-tail, PTRR), after
        # raising the dynamo cache size limit to avoid recompile errors (global side effect).
        # Then checks that a too-short sequence (64 * world_size) raises NotImplementedError:
        # "Q_LEN ... is not divisible" for none/head-tail, and "must be divisible by
        # group_size" for PTRR.
        seq_length_list = [256 * self.world_size, 2048]
        load_balance_type_list = [
            "None",
            "_HeadTailLoadBalancer",
            "_PTRRLoadBalancer",
        ]

        # NOTE: Each (seq_len, load_balance_type) tuple introduces 2
        # create_block_mask compilations: 1 for single-rank flex_attention and 1 for
        # CP flex_attention. In order to avoid the "exceeds_recompile_limit" error,
        # we need to increase the cache_size_limit to 2 * num_of_sub_test_runs which
        # will be the total number of compilations in our test case.
        torch._dynamo.config.cache_size_limit = (len(seq_length_list) + 1) * (
            1 + len(load_balance_type_list)
        )

        for qkv_size, lb_type in itertools.product(
            seq_length_list, load_balance_type_list
        ):
            block_mask = compiled_create_block_mask(
                causal_mask,
                B=1,
                H=1,
                Q_LEN=qkv_size,
                KV_LEN=qkv_size,
                device=self.device_type,
            )
            self._test_cp_flex_attention(
                qkv_size=qkv_size, block_mask=block_mask, lb_type=lb_type
            )

        # NOTE: Context Parallel should not be used for small attentions (block_size < 128)
        qkv_size = 64 * self.world_size
        block_mask = compiled_create_block_mask(
            causal_mask,
            B=1,
            H=1,
            Q_LEN=qkv_size,
            KV_LEN=qkv_size,
            device=self.device_type,
        )

        for lb_type in ["None", "_HeadTailLoadBalancer"]:
            with self.assertRaisesRegex(
                NotImplementedError,
                f"Q_LEN {qkv_size} is not divisible",
            ):
                self._test_cp_flex_attention(
                    qkv_size=qkv_size, block_mask=block_mask, lb_type=lb_type
                )

        for lb_type in ["_PTRRLoadBalancer"]:
            with self.assertRaisesRegex(
                NotImplementedError,
                "must be divisible by group_size",
            ):
                self._test_cp_flex_attention(
                    qkv_size=qkv_size, block_mask=block_mask, lb_type=lb_type
                )

    # TODO: merge with the above test
    @skip_if_lt_x_gpu(2)
    @with_comms
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @skip_if_rocm_ver_atleast_multiprocess([7, 14])
    def test_cp_flex_attention_document_mask(self) -> None:
        # [FUNCTION DOC] Runs the CP flex comparison with causal document masks built from
        # random document lengths (28 documents) for batch sizes 2, 4, 8, sequence lengths
        # (256 * world_size, 2048) and four load balancers (none, head-tail, per-document
        # head-tail, PTRR). Seeds the random module and raises the dynamo cache size limit
        # (global side effects). Per-document balancing uses lengths that are multiples of
        # 2 * world_size.
        random.seed(10)

        # parameters for testing
        doc_count = 28
        batch_size_list = [2, 4, 8]
        max_seq_len_list = [
            256 * self.world_size,
            2048,
            # 128 * self.world_size  # NOTE: Mismatched elements: 8 / 131072 (0.0%),
        ]
        load_balance_type = [
            "None",
            "_HeadTailLoadBalancer",
            "_PerDocumentHeadTailLoadBalancer",
            "_PTRRLoadBalancer",
        ]

        # NOTE: Each (batch_size, seq_len, load_balance_type) tuple introduces 2
        # create_block_mask compilations: 1 for single-rank flex_attention and 1 for
        # CP flex_attention. In order to avoid the "exceeds_recompile_limit" error,
        # we need to increase the cache_size_limit to 2 * num_of_sub_test_runs which
        # will be the total number of compilations in our test case.
        torch._dynamo.config.cache_size_limit = (
            2 * len(batch_size_list) * len(max_seq_len_list) * len(load_balance_type)
        )

        # TODO: change this for-loop to run_subtests
        # Use a for-loop instead of run_subtests because we need to initialize the mask
        # for each subtest. This can be baked into self._test_cp_flex_attention as
        # a str argument denoting mask type.
        for batch_size, max_seq_len, lb_type in itertools.product(
            batch_size_list,
            max_seq_len_list,
            load_balance_type,
        ):
            # initialize document mask
            lengths = [
                (
                    generate_random_lengths_in_chunks(
                        max_seq_len, doc_count, chunk_size=2 * self.world_size
                    )
                    if lb_type == "_PerDocumentHeadTailLoadBalancer"
                    else generate_random_lengths(max_seq_len, doc_count)
                )
                for _ in range(batch_size)
            ]
            offsets = length_to_offsets(lengths, self.device_type)
            document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)
            block_mask = compiled_create_block_mask(
                document_causal_mask,
                B=batch_size,
                H=1,
                Q_LEN=max_seq_len,
                KV_LEN=max_seq_len,
                device=self.device_type,
            )

            self._test_cp_flex_attention(
                qkv_size=max_seq_len,
                B=batch_size,
                lb_type=lb_type,
                block_mask=block_mask,
                document_lengths=lengths,
            )


class TestCPCustomOps(DTensorTestBase):
    # [CLASS DOC] 2-GPU tests of the context-parallel custom operators.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the fixed world size of 2.
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_flex_cp_custom_op(self) -> None:
        # [FUNCTION DOC] Runs torch.library.opcheck on the flex_cp_allgather op with random
        # key/value tensors (without and with requires_grad), gather dim 2 and the name of
        # the cp mesh's process group.
        mesh = init_device_mesh(
            device_type=self.device_type,
            mesh_shape=(self.world_size,),
            mesh_dim_names=("cp",),
        )
        examples_k_v = [
            (
                torch.randn(8, 8, 8, 8, device=self.device_type),
                torch.randn(8, 8, 8, 8, device=self.device_type),
                2,
                c10d._get_process_group_name(mesh.get_group()),
            ),
            (
                torch.randn(8, 8, 8, 8, device=self.device_type, requires_grad=True),
                torch.randn(8, 8, 8, 8, device=self.device_type, requires_grad=True),
                2,
                c10d._get_process_group_name(mesh.get_group()),
            ),
        ]
        for example in examples_k_v:
            torch.library.opcheck(flex_cp_allgather, example)


class TestSharding(DTensorTestBase):
    # [CLASS DOC] 2-GPU tests of _context_parallel_shard (buffer sharding along the sequence
    # dimension with a load balancer) and of DTensor SDPA strategy selection without CP.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the fixed world size of 2.
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_context_parallel_shard(self) -> None:
        # [FUNCTION DOC] Shards a 1-D position tensor and three 2-D tensors with the head-tail
        # load balancer. Asserts each shard has half the sequence and the 1-D shard equals the
        # concatenation of this rank's chunk and the mirrored chunk (of 2 * world_size chunks).
        B = 4
        seq_len = 32

        device_mesh = init_device_mesh(
            mesh_shape=(2,), mesh_dim_names=("cp",), device_type=self.device_type
        )
        freqs_cis = torch.arange(0, seq_len, device=self.device_type)
        q = torch.ones(B * seq_len, device=self.device_type).reshape(B, seq_len)
        k = torch.ones(B * seq_len, device=self.device_type).reshape(B, seq_len)
        v = torch.ones(B * seq_len, device=self.device_type).reshape(B, seq_len)

        load_balancer = _HeadTailLoadBalancer(
            seq_len, self.world_size, torch.device(self.device_type)
        )
        freqs_cis_shard, q_shard, k_shard, v_shard = _context_parallel_shard(
            device_mesh, [freqs_cis, q, k, v], [0, 1, 1, 1], load_balancer=load_balancer
        )
        self.assertEqual(freqs_cis_shard.size(), (seq_len // 2,))
        chunks = freqs_cis.chunk(self.world_size * 2)
        self.assertEqual(
            freqs_cis_shard,
            map_local_tensor_for_rank(
                chunks,
                self.rank,
                lambda chunks, rank: torch.cat(
                    [chunks[rank], chunks[self.world_size * 2 - rank - 1]],
                    dim=0,
                ),
            ),
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_context_parallel_shard_per_document_balance(self) -> None:
        # [FUNCTION DOC] Builds random document lengths (identical on all ranks via a local
        # RNG, each a multiple of 2 * world_size), shards a token-id buffer with the
        # per-document head-tail balancer, and asserts this rank's causal-attention cost
        # (position within document plus one, summed) equals the total cost over world size.
        # End-to-end regression through the real _context_parallel_shard path:
        # rearrange + shard randomized mixed-length documents and confirm each
        # rank receives an equal share of causal attention work.
        # Rank-major balances every rank exactly.
        # Since a balanced split gives every rank total_work / world_size, each
        # rank can check its own local shard without cross-rank communication.
        # NOTE: seed a local RNG so every rank generates identical document
        # lengths; each length must be a multiple of 2 * world_size.
        rng = random.Random(1234)
        doc_lengths = [
            2 * self.world_size * rng.randint(1, 6) for _ in range(rng.randint(1, 5))
        ]
        seq_len = sum(doc_lengths)
        device_mesh = init_device_mesh(
            mesh_shape=(self.world_size,),
            mesh_dim_names=("cp",),
            device_type=self.device_type,
        )
        # cost[t] = causal work of the query at original position t (1..L per doc).
        cost = []
        for length in doc_lengths:
            cost.extend(range(1, length + 1))
        cost = torch.tensor(cost, device=self.device_type)

        # A token-id buffer so each sharded position maps back to its origin.
        buffer = torch.arange(seq_len, device=self.device_type).view(1, seq_len)
        load_balancer = _PerDocumentHeadTailLoadBalancer(
            [doc_lengths], self.world_size, torch.device(self.device_type)
        )
        (sharded,) = _context_parallel_shard(
            device_mesh, [buffer], [1], load_balancer=load_balancer
        )

        local_load = int(cost[sharded.reshape(-1).long()].sum())
        self.assertEqual(local_load, int(cost.sum()) // self.world_size)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_context_parallel_shard_with_positions(self) -> None:
        # [FUNCTION DOC] Shards an expanded (batch broadcast) positions tensor plus q, k, v
        # along dim 1 with the head-tail balancer. Asserts the positions shard has shape
        # (B, seq_len / 2) and equals this rank's chunk concatenated with its mirrored chunk.
        """Test context parallel sharding with expanded batch dimensions.

        This test validates the fix for buffer sharding when the batch dimension
        is created through expand() or view() operations. Before the fix, the
        loop-based torch.index_select approach failed on expanded tensors.
        """
        B = 4
        seq_len = 32

        device_mesh = init_device_mesh(
            mesh_shape=(2,), mesh_dim_names=("cp",), device_type=self.device_type
        )

        # Create positions tensor and expand to add batch dimension
        positions = torch.arange(0, seq_len, device=self.device_type)
        positions = positions.expand(B, seq_len)

        q = torch.ones(B * seq_len, device=self.device_type).reshape(B, seq_len)
        k = torch.ones(B * seq_len, device=self.device_type).reshape(B, seq_len)
        v = torch.ones(B * seq_len, device=self.device_type).reshape(B, seq_len)

        load_balancer = _HeadTailLoadBalancer(
            seq_len, self.world_size, torch.device(self.device_type)
        )

        # positions has seq_dim=1 (same as q, k, v) after expansion
        positions_shard, q_shard, k_shard, v_shard = _context_parallel_shard(
            device_mesh, [positions, q, k, v], [1, 1, 1, 1], load_balancer=load_balancer
        )

        # Verify the sharded positions tensor has correct shape
        self.assertEqual(positions_shard.size(), (B, seq_len // 2))

        # Verify the sharded values match expected chunked and concatenated results
        # For each batch, the positions should be chunked and rearranged
        chunks = positions.chunk(self.world_size * 2, dim=1)
        expected_positions = map_local_tensor_for_rank(
            chunks,
            self.rank,
            lambda chunks, rank: torch.cat(
                [chunks[rank], chunks[self.world_size * 2 - rank - 1]],
                dim=1,
            ),
        )
        self.assertEqual(positions_shard, expected_positions)

    @skip_if_lt_x_gpu(2)
    @with_comms
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Does not support flash nor efficient attention",
    )
    def test_attention_shard_without_cp(self) -> None:
        # [FUNCTION DOC] For each supported SDPA backend, runs SDPA on DTensors sharded on the
        # sequence dim; registers CP sharding rules, runs SDPA once, unregisters them (clearing
        # the cache), and runs again. Asserts the final output placement is not Shard(2) and is
        # replicated or sharded on dim 0 or 1.
        # KNOWN ISSUE: the first SDPA result (with CP rules registered) is overwritten and not
        # checked; only the post-unregister result is asserted.
        """Test that sharding on sequence dimension without CP enabled is not supported."""
        from torch.distributed.tensor import distribute_tensor, Replicate, Shard

        B = 2
        nheads = 4
        seq_len = 256
        dim = 32

        device_mesh = init_device_mesh(
            mesh_shape=(2,), mesh_dim_names=("cp",), device_type=self.device_type
        )

        for backend in backends:
            with sdpa_kernel(backend):
                dtype = torch.bfloat16
                if backend == SDPBackend.EFFICIENT_ATTENTION:
                    dtype = torch.float32
                # Create q, k, v tensors with shape (B, nheads, seq_len, dim)
                q = torch.randn(
                    B, nheads, seq_len, dim, device=self.device_type, dtype=dtype
                )
                k = torch.randn(
                    B, nheads, seq_len, dim, device=self.device_type, dtype=dtype
                )
                v = torch.randn(
                    B, nheads, seq_len, dim, device=self.device_type, dtype=dtype
                )
                q_dt = distribute_tensor(q, device_mesh, [Shard(2)])
                k_dt = distribute_tensor(k, device_mesh, [Shard(2)])
                v_dt = distribute_tensor(v, device_mesh, [Shard(2)])

                register_cp_sharding_rules()
                out = F.scaled_dot_product_attention(q_dt, k_dt, v_dt)
                unregister_cp_sharding_rules(clear_the_cache=True)
                out = F.scaled_dot_product_attention(q_dt, k_dt, v_dt)
                # Run SDPA with sequence-sharded tensors WITHOUT enabling CP
                # Without CP enabled, DTensor should select a different strategy
                # (not sequence-sharded) because Shard(2) strategy is only available with CP

                # Verify the output is NOT sharded on sequence dimension (dim 2)
                # This proves that CP sharding rules were not used
                self.assertNotEqual(
                    out.placements[0],
                    Shard(2),
                    lambda msg: f"{msg}\nPlacement {out.placements}",
                )
                # The output should be replicated or sharded on batch head dimensions.
                self.assertIn(out.placements[0], [Replicate(), Shard(0), Shard(1)])


class TestContextParallelStyle(DTensorTestBase):
    # [CLASS DOC] Tests how _ContextParallel.flex_input_fn handles positional, keyword and
    # extra arguments, with flex_cp_allgather mocked to return scaled key/value.
    """Test suite for _ContextParallel.flex_input_fn argument handling"""

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the fixed world size of 2.
        return 2

    def _create_test_tensors(self):
        # [FUNCTION DOC] HELPER: Returns random query, key, value tensors of shape
        # (2, 4, 128, 64) on the test device.
        """Helper to create test query, key, value tensors"""
        query = torch.randn(2, 4, 128, 64, device=self.device_type)
        key = torch.randn(2, 4, 128, 64, device=self.device_type)
        value = torch.randn(2, 4, 128, 64, device=self.device_type)
        return query, key, value

    def _setup_mock_and_context(self, mock_allgather, key, value):
        # [FUNCTION DOC] HELPER: Configures the mock to return (key times 2, value times 3),
        # builds a FLEX _ContextParallel style with seq_dim 2 and a device mesh over all
        # ranks. Returns (style, mesh, mocked key, mocked value).
        """Helper to setup mock and create CP instance + device mesh"""
        # Setup mock with transformed tensors
        mock_key = key * 2
        mock_value = value * 3
        mock_allgather.return_value = (mock_key, mock_value)

        # Create CP instance and device mesh
        cp_style = _ContextParallel(
            seq_dim=2, attention_type=_ContextParallel.AttentionType.FLEX
        )
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))

        return cp_style, device_mesh, mock_key, mock_value

    @with_comms
    @unittest.mock.patch(
        "torch.distributed.tensor.experimental._context_parallel._attention.flex_cp_allgather"
    )
    def test_flex_input_fn_all_positional(self, mock_allgather):
        # [FUNCTION DOC] flex_input_fn with query, key, value all positional: the allgather
        # mock is called once, three positional args and no kwargs come back, query is
        # unchanged and key/value are the mock's replacements.
        """Test flex_input_fn with all positional arguments"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh, mock_key, mock_value = self._setup_mock_and_context(
            mock_allgather, key, value
        )

        # Call with all positional args
        args = (query, key, value)
        kwargs = {}
        out_args, out_kwargs = cp_style.flex_input_fn(None, args, kwargs, device_mesh)

        # Verify mock called and output structure
        mock_allgather.assert_called_once()
        self.assertEqual(len(out_args), 3)
        self.assertEqual(len(out_kwargs), 0)

        # Verify query unchanged, key/value replaced
        torch.testing.assert_close(out_args[0], query)
        torch.testing.assert_close(out_args[1], mock_key)
        torch.testing.assert_close(out_args[2], mock_value)

    @with_comms
    @unittest.mock.patch(
        "torch.distributed.tensor.experimental._context_parallel._attention.flex_cp_allgather"
    )
    def test_flex_input_fn_all_keyword(self, mock_allgather):
        # [FUNCTION DOC] flex_input_fn with query, key, value all keyword: no positional args
        # come back, the three keys remain in kwargs, query is unchanged and key/value are
        # replaced by the mock's outputs.
        """Test flex_input_fn with all keyword arguments"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh, mock_key, mock_value = self._setup_mock_and_context(
            mock_allgather, key, value
        )

        # Call with all keyword args
        args = ()
        kwargs = {"query": query, "key": key, "value": value}
        out_args, out_kwargs = cp_style.flex_input_fn(None, args, kwargs, device_mesh)

        # Verify mock called and output structure
        mock_allgather.assert_called_once()
        self.assertEqual(len(out_args), 0)
        self.assertIn("query", out_kwargs)
        self.assertIn("key", out_kwargs)
        self.assertIn("value", out_kwargs)

        # Verify query unchanged, key/value replaced
        torch.testing.assert_close(out_kwargs["query"], query)
        torch.testing.assert_close(out_kwargs["key"], mock_key)
        torch.testing.assert_close(out_kwargs["value"], mock_value)

    @with_comms
    @unittest.mock.patch(
        "torch.distributed.tensor.experimental._context_parallel._attention.flex_cp_allgather"
    )
    def test_flex_input_fn_query_positional_kv_keyword(self, mock_allgather):
        # [FUNCTION DOC] flex_input_fn with positional query and keyword key/value: query stays
        # positional and unchanged, key and value stay in kwargs with replaced values.
        """Test with query positional, key/value keyword"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh, mock_key, mock_value = self._setup_mock_and_context(
            mock_allgather, key, value
        )

        # Query positional, key/value keyword
        args = (query,)
        kwargs = {"key": key, "value": value}
        out_args, out_kwargs = cp_style.flex_input_fn(None, args, kwargs, device_mesh)

        # Verify mock called and output structure
        mock_allgather.assert_called_once()
        self.assertEqual(len(out_args), 1)
        torch.testing.assert_close(out_args[0], query)

        # Verify key/value in kwargs and updated
        self.assertIn("key", out_kwargs)
        self.assertIn("value", out_kwargs)
        torch.testing.assert_close(out_kwargs["key"], mock_key)
        torch.testing.assert_close(out_kwargs["value"], mock_value)

    @with_comms
    @unittest.mock.patch(
        "torch.distributed.tensor.experimental._context_parallel._attention.flex_cp_allgather"
    )
    def test_flex_input_fn_qk_positional_v_keyword(self, mock_allgather):
        # [FUNCTION DOC] flex_input_fn with positional query and key and keyword value: two
        # positional args come back (query unchanged, key replaced) and value in kwargs is
        # replaced.
        """Test with query/key positional, value keyword"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh, mock_key, mock_value = self._setup_mock_and_context(
            mock_allgather, key, value
        )

        # Query/key positional, value keyword
        args = (query, key)
        kwargs = {"value": value}
        out_args, out_kwargs = cp_style.flex_input_fn(None, args, kwargs, device_mesh)

        # Verify mock called and output structure
        mock_allgather.assert_called_once()
        self.assertEqual(len(out_args), 2)
        torch.testing.assert_close(out_args[0], query)
        torch.testing.assert_close(out_args[1], mock_key)

        # Verify value in kwargs and updated
        self.assertIn("value", out_kwargs)
        torch.testing.assert_close(out_kwargs["value"], mock_value)

    @with_comms
    @unittest.mock.patch(
        "torch.distributed.tensor.experimental._context_parallel._attention.flex_cp_allgather"
    )
    def test_flex_input_fn_with_extra_args(self, mock_allgather):
        # [FUNCTION DOC] flex_input_fn with positional query, key, value plus score_mod and
        # block_mask, and an extra keyword enable_gqa: five positional args come back with key
        # and value replaced and the others unchanged, and the extra keyword is preserved.
        """Test with mixed positional/keyword and extra arguments"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh, mock_key, mock_value = self._setup_mock_and_context(
            mock_allgather, key, value
        )

        # Mix of positional and keyword with extra args
        def score_mod(q, k, b, h, m, n):
            # [FUNCTION DOC] HELPER: Placeholder score modifier that returns its first argument
            # unchanged; only used as an opaque extra argument.
            return q

        block_mask = None
        args = (query, key, value, score_mod, block_mask)
        kwargs = {"enable_gqa": False}
        out_args, out_kwargs = cp_style.flex_input_fn(None, args, kwargs, device_mesh)

        # Verify mock called and output structure
        mock_allgather.assert_called_once()
        self.assertEqual(len(out_args), 5)
        torch.testing.assert_close(out_args[0], query)
        torch.testing.assert_close(out_args[1], mock_key)
        torch.testing.assert_close(out_args[2], mock_value)
        self.assertEqual(out_args[3], score_mod)
        self.assertEqual(out_args[4], block_mask)

        # Verify extra kwargs unchanged
        self.assertEqual(out_kwargs["enable_gqa"], False)


class TestContextParallelStyleSDPA(DTensorTestBase):
    # [CLASS DOC] Tests how _ContextParallel.sdpa_input_fn converts positional and keyword
    # query/key/value arguments into sequence-sharded DTensors.
    """Test suite for _ContextParallel.sdpa_input_fn argument handling"""

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the fixed world size of 2.
        return 2

    def _create_test_tensors(self):
        # [FUNCTION DOC] HELPER: Returns random query, key, value tensors of shape
        # (2, 4, 128, 64) on the test device.
        """Helper to create test query, key, value tensors"""
        query = torch.randn(2, 4, 128, 64, device=self.device_type)
        key = torch.randn(2, 4, 128, 64, device=self.device_type)
        value = torch.randn(2, 4, 128, 64, device=self.device_type)
        return query, key, value

    def _setup_context(self):
        # [FUNCTION DOC] HELPER: Returns an SDPA _ContextParallel style with seq_dim 2 and a
        # device mesh over all ranks.
        """Helper to create CP instance and device mesh"""
        cp_style = _ContextParallel(
            seq_dim=2, attention_type=_ContextParallel.AttentionType.SDPA
        )
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))
        return cp_style, device_mesh

    @with_comms
    def test_sdpa_input_fn_all_positional(self):
        # [FUNCTION DOC] sdpa_input_fn with three positional tensors returns three positional
        # DTensors with placement Shard(2) and no kwargs.
        """Test sdpa_input_fn with all positional arguments"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh = self._setup_context()

        # Call with all positional args
        args = (query, key, value)
        kwargs = {}
        out_args, out_kwargs = cp_style.sdpa_input_fn(None, args, kwargs, device_mesh)

        # Verify output structure: should be all positional
        self.assertEqual(len(out_args), 3)
        self.assertEqual(len(out_kwargs), 0)

        # Verify all outputs are DTensors
        self.assertIsInstance(out_args[0], DTensor)
        self.assertIsInstance(out_args[1], DTensor)
        self.assertIsInstance(out_args[2], DTensor)

        # Verify DTensors have correct placement (Shard(2) for seq_dim=2)
        from torch.distributed.tensor.placement_types import Shard

        self.assertEqual(out_args[0].placements, [Shard(2)])
        self.assertEqual(out_args[1].placements, [Shard(2)])
        self.assertEqual(out_args[2].placements, [Shard(2)])

    @with_comms
    def test_sdpa_input_fn_all_keyword(self):
        # [FUNCTION DOC] sdpa_input_fn with keyword query/key/value returns no positional args
        # and DTensors with placement Shard(2) under the same keyword names.
        """Test sdpa_input_fn with all keyword arguments"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh = self._setup_context()

        # Call with all keyword args
        args = ()
        kwargs = {"query": query, "key": key, "value": value}
        out_args, out_kwargs = cp_style.sdpa_input_fn(None, args, kwargs, device_mesh)

        # Verify output structure: should be all keyword
        self.assertEqual(len(out_args), 0)
        self.assertIn("query", out_kwargs)
        self.assertIn("key", out_kwargs)
        self.assertIn("value", out_kwargs)

        # Verify all outputs are DTensors
        self.assertIsInstance(out_kwargs["query"], DTensor)
        self.assertIsInstance(out_kwargs["key"], DTensor)
        self.assertIsInstance(out_kwargs["value"], DTensor)

        # Verify DTensors have correct placement
        from torch.distributed.tensor.placement_types import Shard

        self.assertEqual(out_kwargs["query"].placements, [Shard(2)])
        self.assertEqual(out_kwargs["key"].placements, [Shard(2)])
        self.assertEqual(out_kwargs["value"].placements, [Shard(2)])

    @with_comms
    def test_sdpa_input_fn_query_positional_kv_keyword(self):
        # [FUNCTION DOC] sdpa_input_fn with positional query and keyword key/value keeps that
        # structure and converts all three to DTensors with placement Shard(2).
        """Test sdpa_input_fn with query positional, key/value keyword"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh = self._setup_context()

        # Query positional, key/value keyword
        args = (query,)
        kwargs = {"key": key, "value": value}
        out_args, out_kwargs = cp_style.sdpa_input_fn(None, args, kwargs, device_mesh)

        # Verify output structure: query should be positional, rest keyword
        self.assertEqual(len(out_args), 1)
        self.assertIn("key", out_kwargs)
        self.assertIn("value", out_kwargs)

        # Verify all outputs are DTensors
        self.assertIsInstance(out_args[0], DTensor)
        self.assertIsInstance(out_kwargs["key"], DTensor)
        self.assertIsInstance(out_kwargs["value"], DTensor)

        # Verify DTensors have correct placement
        from torch.distributed.tensor.placement_types import Shard

        self.assertEqual(out_args[0].placements, [Shard(2)])
        self.assertEqual(out_kwargs["key"].placements, [Shard(2)])
        self.assertEqual(out_kwargs["value"].placements, [Shard(2)])


RingAttentionTestWithLocalTensor = create_local_tensor_test_class(
    RingAttentionTest,
    skipped_tests=[
        # Need to make attention implementation local tensor friendly, e.g.
        # rewrite "rank local" logic
        "test_ring_attention_sdpa",
        "test_context_parallel_sdpa_short_sequence",
    ],
)

CPFlexAttentionTestWithLocalTensor = create_local_tensor_test_class(
    CPFlexAttentionTest,
    skipped_tests=[
        # Missing support for batched tensors
        "test_cp_flex_attention_causal_mask",
        "test_cp_flex_attention_document_mask",
    ],
)

TestCPCustomOpsWithLocalTensor = create_local_tensor_test_class(
    TestCPCustomOps,
    skipped_tests=[
        # Missing support for fake tensors
        "test_flex_cp_custom_op",
    ],
)

TestShardingWithLocalTensor = create_local_tensor_test_class(
    TestSharding,
    skipped_tests=[
        # Uses Python-scalar per-rank load extraction, which does not apply to
        # the batched local-tensor execution mode.
        "test_context_parallel_shard_per_document_balance",
    ],
)


class PerDocumentHeadTailLoadBalancerTest(TestCase):
    # [CLASS DOC] CPU-only, single-process tests that the per-document head-tail load
    # balancer lays out indices rank-major so contiguous equal shards carry equal causal
    # attention cost, and that restore indices invert the rearrangement.
    """Non-distributed unit tests for the index layout produced by
    ``_PerDocumentHeadTailLoadBalancer``. Context-Parallel shards the rearranged
    sequence into contiguous, equal, per-rank chunks, so the balancer must lay out
    its indices rank-major (each rank's head+tail chunks of *every* document
    grouped together) for the contiguous cut to land on rank boundaries."""

    @staticmethod
    def _causal_cost(doc_lengths: list[int]) -> list[int]:
        # [FUNCTION DOC] HELPER: Returns per-token causal cost: for each document of length L,
        # the values 1 to L, concatenated over documents.
        # Under a document-causal mask, the query at within-document position p
        # attends p + 1 keys, so its cost is 1, 2, ..., L within each document.
        cost: list[int] = []
        for length in doc_lengths:
            cost.extend(range(1, length + 1))
        return cost

    def _per_rank_causal_load(
        self, doc_lengths: list[int], world_size: int
    ) -> list[int]:
        # [FUNCTION DOC] HELPER: Builds a CPU per-document head-tail balancer for one batch
        # row of documents, takes its rearranged index order, splits it into world_size
        # contiguous equal shards, and returns each shard's summed causal cost.
        seq_len = sum(doc_lengths)
        cost = self._causal_cost(doc_lengths)
        lb = _PerDocumentHeadTailLoadBalancer([doc_lengths], world_size, "cpu")
        # Original positions in rearranged order (batch dim removed).
        rearranged = lb._generate_indices()[0].tolist()
        # CP shards the rearranged stream into world_size contiguous equal chunks.
        shard = seq_len // world_size
        return [
            sum(cost[t] for t in rearranged[r * shard : (r + 1) * shard])
            for r in range(world_size)
        ]

    def test_mixed_length_documents_are_balanced(self) -> None:
        # [FUNCTION DOC] Regression test: documents of lengths 4 and 12 on 2 ranks must give
        # both ranks a causal load of 44.
        # Regression test: with mixed-length documents the document-major layout
        # let the contiguous shard cut fall mid-document, giving loads 26 vs 62
        # (2.38x imbalance). A rank-major layout balances them exactly.
        loads = self._per_rank_causal_load([4, 12], world_size=2)
        self.assertEqual(loads, [44, 44])

    def test_random_documents_are_balanced_across_world_sizes(self) -> None:
        # [FUNCTION DOC] For world sizes 2, 3, 4, 8 and 20 random document sets each (lengths
        # multiples of 2 * world_size), asserts all ranks get identical causal load.
        # The rank-major layout makes every rank's causal load identical for any
        # set of document lengths (each a multiple of 2 * world_size) and any
        # world size, since pairing head chunk r with tail chunk (2P-1-r) gives
        # each rank the same per-document load. Exercise this with randomized,
        # mixed-length documents across several world sizes > 2.
        rng = random.Random(2024)
        for world_size in (2, 3, 4, 8):
            for _ in range(20):
                num_docs = rng.randint(1, 6)
                doc_lengths = [
                    2 * world_size * rng.randint(1, 5) for _ in range(num_docs)
                ]
                loads = self._per_rank_causal_load(doc_lengths, world_size)
                self.assertEqual(
                    loads,
                    [loads[0]] * world_size,
                    msg=f"world_size={world_size}, doc_lengths={doc_lengths}",
                )

    def test_restore_indices_invert_rearrange(self) -> None:
        # [FUNCTION DOC] For documents of lengths 4 and 12 on 2 ranks, asserts indexing the
        # rearranged indices with the restore indices yields positions 0 to 15 in order.
        lb = _PerDocumentHeadTailLoadBalancer([[4, 12]], 2, "cpu")
        rearranged = lb._generate_indices()[0]
        restore = lb._generate_indices(restore=True)[0]
        self.assertEqual(rearranged[restore].tolist(), list(range(16)))


if __name__ == "__main__":
    run_tests()
