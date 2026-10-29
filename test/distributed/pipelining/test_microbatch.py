# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Device-generic tests for torch.distributed.pipelining.microbatch: splitting
# args/kwargs into chunks along a batch dimension, merging chunks back, splitting flex-attention
# BlockMask objects (including batch size one and None), and TensorChunkSpec usage with pipeline.
# Seeds the RNG at import time and instantiates MicrobatchTests for cpu, cuda, hpu and xpu.
from model_registry import ModelWithKwargs

import torch
from torch.distributed.pipelining import pipeline
from torch.distributed.pipelining.microbatch import (
    merge_chunks,
    split_args_kwargs_into_chunks,
    TensorChunkSpec,
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCPUIf,
    skipXPUIf,
)
from torch.testing._internal.common_utils import run_tests, TestCase


d_hid = 512
torch.manual_seed(0)


class MicrobatchTests(TestCase):
    # [CLASS DOC] Test case for microbatch splitting and merging utilities; device-generic tests
    # take a device argument supplied by instantiate_device_type_tests.
    def test_split_and_merge(self):
        # [FUNCTION DOC] Splits three tensors (batch 128, 256, 512) given positionally and by
        # keyword into 2 chunks along dim 0 and checks chunk counts and shapes (first dim halved).
        # Then merges the chunks with TensorChunkSpec(0) specs and asserts the originals are
        # recovered. Raises AssertionError on any mismatch.
        x0 = torch.randn(128, d_hid)
        x1 = torch.randn(256, d_hid)
        x2 = torch.randn(512, d_hid)

        args = (x0, x1, x2)
        kwargs = {"x0": x0, "x1": x1, "x2": x2}

        # Default chunking: dim 0
        arg_chunks, kwarg_chunks = split_args_kwargs_into_chunks(args, kwargs, 2)
        if len(arg_chunks) != 2:
            raise AssertionError(f"Expected 2 arg_chunks, got {len(arg_chunks)}")
        if len(kwarg_chunks) != 2:
            raise AssertionError(f"Expected 2 kwarg_chunks, got {len(kwarg_chunks)}")
        if arg_chunks[0][0].shape != torch.Size([64, d_hid]):
            raise AssertionError(
                f"Expected arg_chunks[0][0].shape == [64, {d_hid}], got {arg_chunks[0][0].shape}"
            )
        if arg_chunks[1][0].shape != torch.Size([64, d_hid]):
            raise AssertionError(
                f"Expected arg_chunks[1][0].shape == [64, {d_hid}], got {arg_chunks[1][0].shape}"
            )
        if arg_chunks[0][1].shape != torch.Size([128, d_hid]):
            raise AssertionError(
                f"Expected arg_chunks[0][1].shape == [128, {d_hid}], got {arg_chunks[0][1].shape}"
            )
        if arg_chunks[0][2].shape != torch.Size([256, d_hid]):
            raise AssertionError(
                f"Expected arg_chunks[0][2].shape == [256, {d_hid}], got {arg_chunks[0][2].shape}"
            )
        if kwarg_chunks[0]["x0"].shape != torch.Size([64, d_hid]):
            raise AssertionError(
                f"Expected kwarg_chunks[0]['x0'].shape == [64, {d_hid}], got {kwarg_chunks[0]['x0'].shape}"
            )
        if kwarg_chunks[0]["x1"].shape != torch.Size([128, d_hid]):
            raise AssertionError(
                f"Expected kwarg_chunks[0]['x1'].shape == [128, {d_hid}], got {kwarg_chunks[0]['x1'].shape}"
            )
        if kwarg_chunks[1]["x2"].shape != torch.Size([256, d_hid]):
            raise AssertionError(
                f"Expected kwarg_chunks[1]['x2'].shape == [256, {d_hid}], got {kwarg_chunks[1]['x2'].shape}"
            )

        # Merge chunks back together
        merged_args = merge_chunks(
            arg_chunks,
            (TensorChunkSpec(0), TensorChunkSpec(0), TensorChunkSpec(0)),
        )
        torch.testing.assert_close(merged_args, args)

        merged_kwargs = merge_chunks(
            kwarg_chunks,
            {
                "x0": TensorChunkSpec(0),
                "x1": TensorChunkSpec(0),
                "x2": TensorChunkSpec(0),
            },
        )
        torch.testing.assert_close(merged_kwargs, kwargs)
        print("Microbatch test passed")

    @skipCPUIf(True, "Flex attention backward is not supported on CPU")
    def test_split_block_mask(self, device):
        # [FUNCTION DOC] Builds a document-packed block-causal flex-attention BlockMask (batch 6)
        # and a second identical mask, runs attention forward and backward on the full batch,
        # then splits q, k, v and a dict holding both masks into 4 chunks. Runs attention per
        # chunk with backward, concatenates chunk inputs, grads, outputs and BlockMask index
        # tensors, and asserts they equal the full-batch results. Skipped on CPU. Uses
        # torch.compile on cuda only.
        B = 6
        H = 1
        SEQ_LEN = 512
        DIM = 32
        DOC_LEN = 30

        def create_block_causal_mask(batch, eos_id: int):
            # [FUNCTION DOC] HELPER: Marks document boundaries where tokens equal eos_id (and
            # the last position), computes a per-token document index via cumulative sum, and
            # returns a mask function allowing attention within the same document and causally.
            mask = batch == eos_id
            mask[:, -1] = True
            acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
            seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
            seq_idx[:, 1:] = acc_mask[:, :-1]

            def block_causal_mask(
                b: torch.Tensor,
                h: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ):
                # [FUNCTION DOC] HELPER: Mask predicate: true when query and key are in the
                # same document and the key is not after the query.
                return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

            return block_causal_mask

        # Create a fake batch which packs several documents together, which
        # each has DOC_LEN tokens and the last token is the EOS token, DOC_LEN - 1.
        total_elements = B * SEQ_LEN
        batch = torch.arange(total_elements, device=device) % DOC_LEN
        batch = batch.reshape(B, SEQ_LEN)
        q, k, v = (
            torch.randn(B, H, SEQ_LEN, DIM, device=device, requires_grad=True)
            for i in range(3)
        )
        block_mask_fn = torch.compile(create_block_mask, fullgraph=True)
        block_mask = block_mask_fn(
            create_block_causal_mask(batch, DOC_LEN - 1),
            B=B,
            H=H,
            Q_LEN=SEQ_LEN,
            KV_LEN=SEQ_LEN,
            device=device,
        )
        block_mask2 = block_mask_fn(
            create_block_causal_mask(batch, DOC_LEN - 1),
            B=B,
            H=H,
            Q_LEN=SEQ_LEN,
            KV_LEN=SEQ_LEN,
            device=device,
        )
        if device == "cuda":
            flex_fn = torch.compile(flex_attention)
        else:
            # It's unclear why CPU + torch.compile + flex_attention can cause an issue.
            flex_fn = flex_attention
        out = flex_fn(q, k, v, block_mask=block_mask)
        out.sum().backward()

        q_clone, k_clone, v_clone = (target.clone().detach() for target in (q, k, v))
        arg_split, _ = split_args_kwargs_into_chunks(
            (
                q_clone,
                k_clone,
                v_clone,
                {"unused_block_mask": block_mask2, "block_mask": block_mask},
            ),
            {},
            chunks=4,
            args_chunk_spec=None,
            kwargs_chunk_spec=None,
        )
        if len(arg_split) != 4:
            raise AssertionError(f"Expected 4 arg_split, got {len(arg_split)}")

        q_total_chunks = []
        dq_total_chunks = []
        k_total_chunks = []
        dk_total_chunks = []
        v_total_chunks = []
        dv_total_chunks = []
        block_mask_total_chunks = []
        out_total_chunks = []
        for i in range(len(arg_split)):
            q_chunk, k_chunk, v_chunk, block_mask_chunk = arg_split[i]
            for chunk, total_chunks in zip(
                (q_chunk, k_chunk, v_chunk),
                (q_total_chunks, k_total_chunks, v_total_chunks),
            ):
                chunk.requires_grad = True
                total_chunks.append(chunk)

            out_chunk = flex_fn(
                q_chunk, k_chunk, v_chunk, block_mask=block_mask_chunk["block_mask"]
            )

            out_chunk.sum().backward()
            dq_total_chunks.append(q_chunk.grad)
            dk_total_chunks.append(k_chunk.grad)
            dv_total_chunks.append(v_chunk.grad)
            block_mask_total_chunks.append(block_mask_chunk["block_mask"])
            out_total_chunks.append(out_chunk)

        concat_q = torch.cat(q_total_chunks, dim=0)
        concat_dq = torch.cat(dq_total_chunks, dim=0)
        concat_k = torch.cat(k_total_chunks, dim=0)
        concat_dk = torch.cat(dk_total_chunks, dim=0)
        concat_v = torch.cat(v_total_chunks, dim=0)
        concat_dv = torch.cat(dv_total_chunks, dim=0)
        concat_kv_indices = torch.cat(
            [bm.kv_indices for bm in block_mask_total_chunks], dim=0
        )
        concat_kv_num_blocks = torch.cat(
            [bm.kv_num_blocks for bm in block_mask_total_chunks], dim=0
        )
        concat_kv_full_num_blocks = torch.cat(
            [bm.full_kv_num_blocks for bm in block_mask_total_chunks], dim=0
        )
        concat_kv_full_indices = torch.cat(
            [bm.full_kv_indices for bm in block_mask_total_chunks], dim=0
        )
        concat_out = torch.cat(out_total_chunks, dim=0)
        self.assertEqual(concat_q, q)
        self.assertEqual(concat_dq, q.grad)
        self.assertEqual(concat_k, k)
        self.assertEqual(concat_dk, k.grad)
        self.assertEqual(concat_v, v)
        self.assertEqual(concat_dv, v.grad)
        self.assertEqual(concat_kv_indices, block_mask.kv_indices)
        self.assertEqual(concat_kv_num_blocks, block_mask.kv_num_blocks)
        self.assertEqual(concat_kv_full_num_blocks, block_mask.full_kv_num_blocks)
        self.assertEqual(concat_kv_full_indices, block_mask.full_kv_indices)
        self.assertEqual(concat_out, out)

    def test_split_block_mask_batch_size_one(self, device):
        # [FUNCTION DOC] Creates a causal BlockMask with batch dimension 1 for inputs of batch
        # 6, splits the inputs and mask dict into 4 chunks, and checks the BlockMask is
        # broadcast to each chunk: concatenated per-chunk attention outputs must equal the
        # full-batch output. Raises AssertionError if the number of chunks is not 4.
        B = 6
        H = 1
        SEQ_LEN = 512
        DIM = 32

        def create_causal_mask():
            # [FUNCTION DOC] HELPER: Returns a plain causal mask predicate function.
            def causal_mask(
                b: torch.Tensor,
                h: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ):
                # [FUNCTION DOC] HELPER: True when the key index is at or before the query.
                return q_idx >= kv_idx

            return causal_mask

        q, k, v = (torch.randn(B, H, SEQ_LEN, DIM, device=device) for i in range(3))
        block_mask_fn = torch.compile(create_block_mask, fullgraph=True)
        block_mask = block_mask_fn(
            create_causal_mask(),
            B=1,
            H=H,
            Q_LEN=SEQ_LEN,
            KV_LEN=SEQ_LEN,
            device=device,
        )
        if device == "cuda":
            flex_fn = torch.compile(flex_attention)
        else:
            # It's unclear why CPU + torch.compile + flex_attention can cause an issue.
            flex_fn = flex_attention
        out = flex_fn(q, k, v, block_mask=block_mask)

        q_clone, k_clone, v_clone = (target.clone().detach() for target in (q, k, v))
        arg_split, _ = split_args_kwargs_into_chunks(
            (q_clone, k_clone, v_clone, {"block_mask": block_mask}),
            {},
            chunks=4,
            args_chunk_spec=None,
            kwargs_chunk_spec=None,
        )

        if len(arg_split) != 4:
            raise AssertionError(f"Expected 4 arg_split, got {len(arg_split)}")

        out_total_chunks = []
        for i in range(len(arg_split)):
            q_chunk, k_chunk, v_chunk, block_mask_chunk = arg_split[i]
            out_chunk = flex_fn(
                q_chunk, k_chunk, v_chunk, block_mask=block_mask_chunk["block_mask"]
            )
            out_total_chunks.append(out_chunk)

        concat_out = torch.cat(out_total_chunks, dim=0)
        self.assertEqual(concat_out, out)

    def test_split_block_mask_none(self, device):
        # [FUNCTION DOC] Splits q, k, v plus a None positional arg and a None keyword arg into 4
        # chunks and asserts the chunk count and that the None values are passed through
        # unchanged in every chunk.
        B = 6
        H = 1
        SEQ_LEN = 512
        DIM = 32

        q, k, v = (torch.randn(B, H, SEQ_LEN, DIM, device=device) for i in range(3))
        arg_split, kwarg_split = split_args_kwargs_into_chunks(
            (q, k, v, None),
            {"attention_mask": None},
            chunks=4,
            args_chunk_spec=None,
            kwargs_chunk_spec=None,
        )

        if len(arg_split) != 4:
            raise AssertionError(f"Expected 4 arg_split, got {len(arg_split)}")

        for i in range(len(arg_split)):
            self.assertIsNone(arg_split[i][3])
            self.assertIsNone(kwarg_split[i]["attention_mask"])

    @skipXPUIf(True, "https://github.com/intel/torch-xpu-ops/issues/1682")
    def test_chunk_spec(self, device):
        # [FUNCTION DOC] Splits an input x (positional, chunk spec dim 0) and y (keyword, chunk
        # spec dim 0) into 4 chunks using TensorChunkSpec.from_tuple/from_dict, traces
        # ModelWithKwargs with pipeline using the first chunk as example inputs, and asserts the
        # pipe output on the full inputs matches eager output. Skipped on XPU.
        mod = ModelWithKwargs().to(device)
        batch_size = ModelWithKwargs.DEFAULT_BATCH_SIZE

        x = torch.randn(batch_size, d_hid, device=device)
        y = torch.randn(batch_size, d_hid, device=device)

        num_chunks = 4

        args_chunk_spec = TensorChunkSpec.from_tuple((0,))
        kwargs_chunk_spec = TensorChunkSpec.from_dict({"y": 0})

        args_split, kwargs_split = split_args_kwargs_into_chunks(
            (x,),
            {"y": y},
            num_chunks,
            args_chunk_spec,
            kwargs_chunk_spec,
        )

        pipe = pipeline(
            mod,
            mb_args=args_split[0],
            mb_kwargs=kwargs_split[0],
        ).to(device)

        ref = mod(x, y)
        out = pipe(x, y)[0]

        torch.testing.assert_close(out, ref)
        print(f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")


devices = ["cpu", "cuda", "hpu", "xpu"]
instantiate_device_type_tests(
    MicrobatchTests, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
