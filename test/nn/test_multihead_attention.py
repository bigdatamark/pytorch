# Owner(s): ["module: nn"]
# [MODULE DOC] Tests for torch.nn.MultiheadAttention. Contains a CPU test class that compares
# the module and the functional multi_head_attention_forward against a numpy reference
# implementation and checks shape-validation errors, plus a device-generic class covering the
# fast path (BetterTransformer), dtypes and bias/weight edge cases. Import-time side effects:
# optionally imports numpy, instantiates the device-type tests into module globals and
# instantiates the parametrized tests.
import contextlib
import random
import unittest
import unittest.mock as mock

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyOn,
)
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize as parametrize_test,
    run_tests,
    TEST_CUDA,
    TEST_NUMPY,
    TEST_WITH_CROSSREF,
    xfailIfNoAcceleratorTriton,
)


if TEST_NUMPY:
    import numpy as np


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.py to list it, otherwise it will NOT be run in
# CI.


class TestMultiheadAttentionNN(NNTestCase):
    # [CLASS DOC] CPU-oriented NNTestCase for MultiheadAttention. When CUDA is available it
    # turns on the class-level CUDA memory leak check and non-default-stream flags. No
    # instance state; each test builds its own module and tensors.
    if TEST_CUDA:
        _do_cuda_memory_leak_check = True
        _do_cuda_non_default_stream = True

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    @parametrize_test("average_attn_weights", [True, False])
    def test_multihead_attention(self, average_attn_weights):
        # [FUNCTION DOC] Parametrized over average_attn_weights. Defines numpy reference
        # helpers (scaled dot attention, batched matmul, softmax, head split/combine, linear
        # map) and a nested driver that repeats 100 random trials, builds a
        # MultiheadAttention, runs the functional forward and compares output and attention
        # weights with the numpy reference using an absolute tolerance. Then runs nested
        # scenarios (bias_kv, zero_attn, key padding mask, saved kv, same embed dim). The
        # scenario mixing bias_kv with static kv is expected to raise an AssertionError about
        # bias on a static key. Skipped without numpy.
        def _scaled_dot_attn_ref(
            Q,
            K,
            V,
            dims,
            unseen_mask=None,
            key_padding_mask=None,
            average_attn_weights=average_attn_weights,
        ):
            # [FUNCTION DOC] HELPER: Numpy reference for scaled dot product attention over 4-D
            # arrays. Divides keys by the square root of the head size, multiplies queries by
            # transposed keys, sets masked positions to negative infinity (positions where the
            # unseen mask is zero or the key padding mask is set), applies softmax and
            # multiplies by values. Returns the attended values and the attention weights,
            # averaged over heads when average_attn_weights is true.
            """Numpy-based reference implementation of scaled dot attention
            for testing"""

            QKT = _batchmatmul(
                Q,
                np.transpose(K, axes=[0, 1, 3, 2])
                / np.sqrt(dims[3], dtype=np.float32),  # divide by sqrt(d_head)
            )
            b1, b2, s1, s2 = QKT.shape
            if unseen_mask is not None or key_padding_mask is not None:
                # assert s1 == s2
                for i in range(b1):
                    for j in range(b2):
                        for m in range(s1):
                            for n in range(s2):
                                if unseen_mask is not None and unseen_mask[m][n] == 0:
                                    QKT[i, j, m, n] = -np.inf
                                if (
                                    key_padding_mask is not None
                                    and key_padding_mask[i][n]
                                ):
                                    QKT[i, j, m, n] = -np.inf

            reference = _softmax(QKT)
            ref_attn_weight = reference
            if average_attn_weights:
                ref_attn_weight = np.sum(ref_attn_weight, axis=1) / b2
            reference = _batchmatmul(reference, V)
            return reference, ref_attn_weight

        def _batchmatmul(a, b):  # batchmatmul over 4 dim matrix
            # [FUNCTION DOC] HELPER: Numpy batched matrix multiply of two 4-D arrays. Raises
            # AssertionError if the first two dimensions differ. Loops over batch and head
            # indices and returns a float32 array.
            """Numpy-based batch matrix multiply over 4 dim matrix"""
            if a.shape[0] != b.shape[0]:
                raise AssertionError(
                    f"Expected a.shape[0] == b.shape[0], got {a.shape[0]} vs {b.shape[0]}"
                )
            if a.shape[1] != b.shape[1]:
                raise AssertionError(
                    f"Expected a.shape[1] == b.shape[1], got {a.shape[1]} vs {b.shape[1]}"
                )
            retval = np.zeros(
                (a.shape[0], a.shape[1], a.shape[2], b.shape[3]), dtype=np.float32
            )
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    retval[i, j, :, :] = np.matmul(a[i, j, :, :], b[i, j, :, :])
            return retval

        def _softmax(x):  # softmax over 4 dim matrix
            # [FUNCTION DOC] HELPER: Numpy softmax over the last axis of a 4-D array, computed
            # with a loop per row with max subtraction for stability. Silences numpy
            # invalid-value warnings globally via seterr. Returns a float64 array; fully
            # masked rows yield nan.
            """Numpy-based reference softmax over 4 dim matrix"""
            np.seterr(invalid="ignore")
            output = np.zeros(x.shape, dtype=np.float64)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        x_curr = x[i, j, k, :]
                        e_x = np.exp(x_curr - np.amax(x_curr))
                        output[i, j, k, :] = e_x / np.sum(e_x)
            return output

        def _split_heads_ref(X, dims, nheads, d_head):
            # [FUNCTION DOC] HELPER: Reference head split. Reshapes a (batch, seq, features)
            # array into heads and transposes so the result has shape batch, heads, seq, head
            # size. The dims argument is a list whose first two entries are batch and seq.
            X_split = np.reshape(X, dims[:2] + [nheads, d_head])
            X_split_transposed = np.transpose(X_split, [0, 2, 1, 3])
            reference = np.reshape(
                X_split_transposed, [dims[0], nheads, dims[1], d_head]
            )
            return reference

        def _combine_heads_ref(X, dims, nheads, d_head):
            # [FUNCTION DOC] HELPER: Reference head merge. Transposes a (batch, heads, seq,
            # head size) array back and reshapes it to batch, seq, heads times head size.
            X_transposed = np.transpose(X, [0, 2, 1, 3])
            reference = np.reshape(X_transposed, dims[:2] + [nheads * d_head])
            return reference

        def _fc(X, X_weight, X_bias):
            # [FUNCTION DOC] HELPER: Applies a linear layer in numpy: multiplies the input by
            # the transposed weight and adds the bias; weight and bias are detached torch
            # tensors converted to numpy.
            X_fc_b = X_bias.detach().numpy()
            X_fc_w = X_weight.detach().numpy()
            return np.matmul(X, np.transpose(X_fc_w)) + X_fc_b

        def _create_src_lengths_mask(batch_size, src_lengths):
            # [FUNCTION DOC] HELPER: Builds an integer mask of shape batch size by maximum
            # source length that is 1 where the position index is below the sequence length.
            # Not called anywhere in this test (dead code).
            """
            Generate boolean mask to prevent attention beyond the end of source
            Inputs:
              batch_size : int
              src_lengths : [batch_size] of sentence lengths
            Outputs:
              [batch_size, max_src_len]
            """
            max_srclen = src_lengths.max()
            src_indices = torch.arange(0, max_srclen).unsqueeze(0).to(src_lengths)
            src_indices = src_indices.expand(batch_size, max_srclen)
            src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_srclen)
            # returns [batch_size, max_seq_len]
            return (src_indices < src_lengths).int().detach()

        def _multihead_attn_test_helper(
            add_key_padding_mask=False,
            add_bias_kv=False,
            add_zero_attn=False,
            saved_kv=False,
            same_embed_dim=False,
            average_attn_weights=average_attn_weights,
        ):
            # [FUNCTION DOC] HELPER: Randomized comparison driver. For 100 trials picks random
            # batch, sequence, head count and head size, optionally uses a different key/value
            # dimension, saved static k/v, key padding mask, bias_kv and zero_attn, builds a
            # MultiheadAttention and calls the functional forward with either the packed or
            # the separate projection weights. Recomputes the same result with the numpy
            # reference and asserts output shape and values (tolerance 1e-5) and attention
            # weight shape and values. Raises AssertionError on mismatch.
            for _ in range(100):
                batch_sz, seq_len = (random.randint(2, 10) for r in range(2))
                d_head = random.randint(3, 10)
                nheads = random.randint(2, 5) * 2
                d_model = d_head * nheads
                if same_embed_dim:
                    kv_dim = d_model
                else:
                    kv_dim = random.randint(5, 20)
                dims = [batch_sz, seq_len, kv_dim]

                saved_k = None
                saved_k_tensor = None
                saved_v = None
                saved_v_tensor = None
                if saved_kv:
                    saved_k = np.random.rand(batch_sz * nheads, seq_len, d_head)
                    saved_k_tensor = torch.from_numpy(saved_k).to(
                        torch.get_default_dtype()
                    )
                    saved_v = np.random.rand(batch_sz * nheads, seq_len, d_head)
                    saved_v_tensor = torch.from_numpy(saved_v).to(
                        torch.get_default_dtype()
                    )

                key_padding_mask = None
                key_padding_mask_tensor = None
                if add_key_padding_mask:
                    seq_mask = np.random.randint(0, 2, (1, seq_len))
                    key_padding_mask = np.repeat(seq_mask, batch_sz, axis=0) == 1
                    key_padding_mask_tensor = torch.from_numpy(key_padding_mask)
                decoder_state = np.random.rand(batch_sz, d_model)
                K = np.random.rand(*dims)
                V = K
                Q = np.expand_dims(decoder_state, 1)
                attn_mask = np.random.randint(0, 2, size=(1, seq_len))
                attn_mask_tensor = torch.from_numpy(attn_mask).float()
                attn_mask_tensor.masked_fill_(attn_mask_tensor == 0, float("-inf"))
                attn_mask_tensor.masked_fill_(attn_mask_tensor > 0, float("0.0"))

                decoder_state_tensor = torch.from_numpy(decoder_state).to(
                    torch.get_default_dtype()
                )
                source_hid_tensor = (
                    torch.from_numpy(K).to(torch.get_default_dtype()).transpose(0, 1)
                )

                multihead_attn_module = MultiheadAttention(
                    d_model,
                    nheads,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    kdim=kv_dim,
                    vdim=kv_dim,
                )

                if add_bias_kv:
                    bias_k = multihead_attn_module.bias_k.detach().numpy()
                    bias_v = multihead_attn_module.bias_v.detach().numpy()
                else:
                    bias_k = None
                    bias_v = None

                _Q = decoder_state_tensor.unsqueeze(1).transpose(0, 1)
                _V = source_hid_tensor
                _K = source_hid_tensor

                if multihead_attn_module._qkv_same_embed_dim:
                    (
                        result,
                        result_weight,
                    ) = torch.nn.functional.multi_head_attention_forward(
                        _Q,
                        _K,
                        _V,
                        d_model,
                        nheads,
                        multihead_attn_module.in_proj_weight,
                        multihead_attn_module.in_proj_bias,
                        multihead_attn_module.bias_k,
                        multihead_attn_module.bias_v,
                        multihead_attn_module.add_zero_attn,
                        multihead_attn_module.dropout,
                        multihead_attn_module.out_proj.weight,
                        multihead_attn_module.out_proj.bias,
                        multihead_attn_module.training,
                        key_padding_mask_tensor,
                        True,
                        attn_mask_tensor,
                        static_k=saved_k_tensor,
                        static_v=saved_v_tensor,
                        average_attn_weights=average_attn_weights,
                        is_causal=False,
                    )
                else:
                    (
                        result,
                        result_weight,
                    ) = torch.nn.functional.multi_head_attention_forward(
                        _Q,
                        _K,
                        _V,
                        d_model,
                        nheads,
                        None,
                        multihead_attn_module.in_proj_bias,
                        multihead_attn_module.bias_k,
                        multihead_attn_module.bias_v,
                        multihead_attn_module.add_zero_attn,
                        multihead_attn_module.dropout,
                        multihead_attn_module.out_proj.weight,
                        multihead_attn_module.out_proj.bias,
                        multihead_attn_module.training,
                        key_padding_mask_tensor,
                        True,
                        attn_mask_tensor,
                        True,
                        multihead_attn_module.q_proj_weight,
                        multihead_attn_module.k_proj_weight,
                        multihead_attn_module.v_proj_weight,
                        static_k=saved_k_tensor,
                        static_v=saved_v_tensor,
                        average_attn_weights=average_attn_weights,
                        is_causal=False,
                    )

                result = result.squeeze(0).detach().numpy()

                if multihead_attn_module._qkv_same_embed_dim:
                    q_proj_weight = multihead_attn_module.in_proj_weight[:d_model]
                    k_proj_weight = multihead_attn_module.in_proj_weight[
                        d_model : (d_model * 2)
                    ]
                    v_proj_weight = multihead_attn_module.in_proj_weight[
                        (d_model * 2) :
                    ]
                else:
                    q_proj_weight = multihead_attn_module.q_proj_weight
                    k_proj_weight = multihead_attn_module.k_proj_weight
                    v_proj_weight = multihead_attn_module.v_proj_weight

                Q_fc = _fc(
                    Q, q_proj_weight, multihead_attn_module.in_proj_bias[:d_model]
                )
                K_fc = _fc(
                    K,
                    k_proj_weight,
                    multihead_attn_module.in_proj_bias[d_model : (d_model * 2)],
                )
                V_fc = _fc(
                    V,
                    v_proj_weight,
                    multihead_attn_module.in_proj_bias[(d_model * 2) :],
                )

                if add_bias_kv:
                    K_fc = np.concatenate(
                        (K_fc, np.repeat(bias_k, K_fc.shape[0], axis=0)), axis=1
                    )
                    V_fc = np.concatenate(
                        (V_fc, np.repeat(bias_v, V_fc.shape[0], axis=0)), axis=1
                    )
                    if attn_mask is not None:
                        attn_mask = np.concatenate((attn_mask, np.ones([1, 1])), axis=1)
                    if key_padding_mask is not None:
                        key_padding_mask = np.concatenate(
                            (
                                key_padding_mask,
                                np.full((batch_sz, 1), False, dtype=bool),
                            ),
                            axis=1,
                        )
                    dims[1] += 1
                Q_split = _split_heads_ref(Q_fc, [batch_sz, 1, d_model], nheads, d_head)

                if saved_k is not None:
                    K_split = np.reshape(saved_k, [dims[0], nheads, dims[1], d_head])
                else:
                    K_split = _split_heads_ref(K_fc, dims, nheads, d_head)

                if saved_v is not None:
                    V_split = np.reshape(saved_v, [dims[0], nheads, dims[1], d_head])
                else:
                    V_split = _split_heads_ref(V_fc, dims, nheads, d_head)

                if add_zero_attn:
                    dims[1] += 1
                    K_split = np.concatenate(
                        (
                            K_split,
                            np.zeros(
                                [
                                    K_split.shape[0],
                                    K_split.shape[1],
                                    1,
                                    K_split.shape[3],
                                ]
                            ),
                        ),
                        axis=2,
                    )
                    V_split = np.concatenate(
                        (
                            V_split,
                            np.zeros(
                                [
                                    V_split.shape[0],
                                    V_split.shape[1],
                                    1,
                                    V_split.shape[3],
                                ]
                            ),
                        ),
                        axis=2,
                    )

                    if attn_mask is not None:
                        attn_mask = np.concatenate((attn_mask, np.ones([1, 1])), axis=1)

                    if key_padding_mask is not None:
                        key_padding_mask = np.concatenate(
                            (
                                key_padding_mask,
                                np.full((batch_sz, 1), False, dtype=bool),
                            ),
                            axis=1,
                        )
                attn_heads, ref_attn_weight = _scaled_dot_attn_ref(
                    Q=Q_split,
                    K=K_split,
                    V=V_split,
                    dims=Q_split.shape,
                    unseen_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
                combined_attn_heads = _combine_heads_ref(
                    X=attn_heads, dims=[batch_sz, 1], nheads=nheads, d_head=d_head
                )

                reference = _fc(
                    combined_attn_heads,
                    multihead_attn_module.out_proj.weight,
                    multihead_attn_module.out_proj.bias,
                )
                reference = np.squeeze(reference, axis=1)

                # result = reference
                self.assertEqual(tuple(result.shape), (batch_sz, d_model))
                np.testing.assert_allclose(result, reference, atol=1e-5)

                # result_weight = ref_attn_weight
                result_weight = result_weight.detach().numpy()
                self.assertEqual(
                    tuple(result_weight.shape), tuple(ref_attn_weight.shape)
                )
                np.testing.assert_allclose(result_weight, ref_attn_weight, atol=1e-5)

        def test_multihead_attn_add_bias_kv():
            # [FUNCTION DOC] HELPER: Runs the comparison driver with bias_kv enabled.
            _multihead_attn_test_helper(add_bias_kv=True)

        def test_multihead_attn_add_zero_attn():
            # [FUNCTION DOC] HELPER: Runs the comparison driver with zero_attn enabled.
            _multihead_attn_test_helper(add_zero_attn=True)

        def test_multihead_attn_no_masking():
            # [FUNCTION DOC] HELPER: Runs the comparison driver with no optional features.
            _multihead_attn_test_helper()

        def test_multihead_attn_key_padding_mask():
            # [FUNCTION DOC] HELPER: Runs the comparison driver with a random key padding
            # mask.
            _multihead_attn_test_helper(add_key_padding_mask=True)

        def test_multihead_attn_saved_kv():
            # [FUNCTION DOC] HELPER: Runs the comparison driver with saved static key and
            # value tensors.
            _multihead_attn_test_helper(saved_kv=True)

        def test_multihead_attn_add_bias_kv_zero_attn():
            # [FUNCTION DOC] HELPER: Runs the comparison driver with key padding mask, bias_kv
            # and zero_attn together.
            _multihead_attn_test_helper(
                add_key_padding_mask=True, add_bias_kv=True, add_zero_attn=True
            )

        def test_multihead_attn_all_arguments1():
            # [FUNCTION DOC] HELPER: Runs the comparison driver with key padding mask,
            # zero_attn and saved kv.
            _multihead_attn_test_helper(
                add_key_padding_mask=True, add_zero_attn=True, saved_kv=True
            )

        def test_multihead_attn_all_arguments2():
            # [FUNCTION DOC] HELPER: Runs the comparison driver with key padding mask,
            # bias_kv, zero_attn and saved kv. The caller expects this to raise AssertionError
            # because bias cannot be added to a static key.
            _multihead_attn_test_helper(
                add_key_padding_mask=True,
                add_bias_kv=True,
                add_zero_attn=True,
                saved_kv=True,
            )

        def test_multihead_attn_all_arguments3():
            # [FUNCTION DOC] HELPER: Runs the comparison driver with key padding mask,
            # zero_attn, saved kv and equal embedding dimensions for key and value.
            _multihead_attn_test_helper(
                add_key_padding_mask=True,
                add_zero_attn=True,
                saved_kv=True,
                same_embed_dim=True,
            )

        test_multihead_attn_add_zero_attn()  # Test MultiheadAttention with add_zero_attn
        test_multihead_attn_add_bias_kv()  # Test MultiheadAttention with add_bias_kv
        test_multihead_attn_no_masking()  # Test MultiheadAttention without masking
        test_multihead_attn_key_padding_mask()  # Test MultiheadAttention with src lengths
        test_multihead_attn_saved_kv()  # Test MultiheadAttention with static kv.
        test_multihead_attn_add_bias_kv_zero_attn()  # Test MultiheadAttention with bias_kv and zero_attn.
        test_multihead_attn_all_arguments1()  # Test MultiheadAttention with all the argument.
        with self.assertRaisesRegex(
            AssertionError, "bias cannot be added to static key."
        ):
            test_multihead_attn_all_arguments2()  # Test MultiheadAttention with all the argument.
        test_multihead_attn_all_arguments3()  # Test MultiheadAttention with all the argument.

    def test_multihead_attn_3d_attn_mask(self):
        # [FUNCTION DOC] Checks that a 3-D attention mask (one mask per batch and head) gives
        # the same output as running each batch element separately with its own 2-D mask.
        # Masks hold negative infinity or zero; the 3-D mask is built by repeating each batch
        # mask once per head.
        embed_dim = 8
        num_heads = 4
        batch_size = 8
        src_len = 3
        tgt_len = 2

        query = torch.rand(batch_size, tgt_len, embed_dim)  # [N, T, D]
        key = torch.rand(batch_size, src_len, embed_dim)  # [N, S, D]
        value = key  # [N, S, D]
        attn_mask = torch.randint(
            0, 2, (batch_size, tgt_len, src_len)
        ).float()  # [N, T, S]
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf")).masked_fill(
            attn_mask == 1, 0.0
        )

        mta_model = torch.nn.MultiheadAttention(embed_dim, num_heads)

        # Generate 3D results
        attn_mask_3d = torch.repeat_interleave(
            attn_mask, num_heads, dim=0
        )  # [N * H, T, S]
        output_3d = mta_model(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            attn_mask=attn_mask_3d,
        )[0]
        output_3d = output_3d.transpose(0, 1)  # [N, T, D]

        for i in range(batch_size):
            output_2d = mta_model(
                query[i].unsqueeze(0).transpose(0, 1),
                key[i].unsqueeze(0).transpose(0, 1),
                value[i].unsqueeze(0).transpose(0, 1),
                attn_mask=attn_mask[i],
            )[0]

            # output_2d in shape of [T, 1, D]
            self.assertEqual(output_3d[i].unsqueeze(0).transpose(0, 1), output_2d)

    def test_multihead_attn_no_bias(self):
        # [FUNCTION DOC] Checks that constructing MultiheadAttention with bias disabled leaves
        # both the input projection bias and the output projection bias as None.
        embed_dim = 8
        num_heads = 4
        mha = torch.nn.MultiheadAttention(embed_dim, num_heads, bias=False)

        # Verify that bias=False applies to both in and out projection layers.
        self.assertIsNone(mha.in_proj_bias)
        self.assertIsNone(mha.out_proj.bias)

    def _test_multihead_attn_invalid_shape_impl(self, mha):
        # [FUNCTION DOC] HELPER: Shared body for the invalid shape tests. For the given
        # module, calls it with 3-D (batched) and 2-D (unbatched) inputs that have a wrongly
        # shaped key, value, key padding mask or attention mask, and asserts each call raises
        # AssertionError with the expected message about expected dimensionality or the
        # expected 3-D attention mask shape.
        # Batched (3D) query cases
        query = torch.randn(4, 4, 4)
        key = torch.randn(4, 4, 4)
        value = torch.randn(4, 4, 4)

        msg = "expected `key` and `value` to be 3-D but found 2-D and 3-D tensors respectively"
        # 3D query, 2D key and 3D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, torch.randn(4, 4), value)

        msg = "expected `key` and `value` to be 3-D but found 3-D and 2-D tensors respectively"
        # 3D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, torch.randn(4, 4))

        msg = "expected `key_padding_mask` to be `None` or 2-D but found 1-D tensor instead"
        # 3D query, 3D key, 3D value and 1D key_padding_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                key_padding_mask=torch.tensor(
                    [False, False, True, True], dtype=torch.bool
                ),
            )

        msg = (
            "expected `attn_mask` to be `None`, 2-D or 3-D but found 1-D tensor instead"
        )
        # 3D query, 3D key, 3D value and 1D attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                attn_mask=torch.tensor([False, False, True, True], dtype=torch.bool),
            )

        # Unbatched (2D) query cases
        query = torch.randn(4, 4)
        key = torch.randn(4, 4)
        value = torch.randn(4, 4)

        msg = "expected `key` and `value` to be 2-D but found 3-D and 2-D tensors respectively"
        # 2D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, torch.randn(4, 4, 4), value)

        msg = "expected `key` and `value` to be 2-D but found 2-D and 3-D tensors respectively"
        # 2D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, torch.randn(4, 4, 4))

        msg = "expected `key_padding_mask` to be `None` or 1-D but found 2-D tensor instead"
        # 2D query, 2D key, 2D value and 1D key_padding_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                key_padding_mask=torch.tensor(
                    [[False, False, True, True] * 2], dtype=torch.bool
                ),
            )

        msg = (
            "expected `attn_mask` to be `None`, 2-D or 3-D but found 1-D tensor instead"
        )
        # 2D query, 2D key, 2D value and 1D attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                attn_mask=torch.tensor([False, False, True, True], dtype=torch.bool),
            )

        msg = r"Expected `attn_mask` shape to be \(4, 4, 4\)"
        # 2D query, 2D key, 2D value and 3D incorrect attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                attn_mask=torch.randn(5, 4, 4).bernoulli_().to(torch.bool),
            )

    def test_multihead_attn_invalid_shape(self):
        # [FUNCTION DOC] Runs the shared invalid shape checks on a fresh module in training
        # mode, and again on the same module in eval mode under no_grad.
        mha = torch.nn.MultiheadAttention(4, 4)
        self._test_multihead_attn_invalid_shape_impl(mha)
        # Give the test a chance to hit the fast path. (Right now, it
        # won't, but gating may be less restricted in the future.)
        with torch.no_grad():
            self._test_multihead_attn_invalid_shape_impl(mha.eval())

    @torch.no_grad()
    def test_multihead_attn_fast_path_invalid_shape(self):
        # [FUNCTION DOC] Same invalid shape checks for an eval-mode batch_first module under
        # no_grad with need_weights disabled, where the fast path is eligible. Batched cases
        # fall back to the slow path and give the usual messages; the unbatched cases give the
        # regular-path messages because the fast path does not support 2-D input.
        mha = torch.nn.MultiheadAttention(4, 4, batch_first=True).eval()

        # Batched (3D) query cases
        query = torch.randn(4, 4, 4)
        key = torch.randn(4, 4, 4)
        value = torch.randn(4, 4, 4)

        # Currently, this case will just go to the slow path and get
        # the usual message because it fails the requirement to be
        # batched.
        msg = "expected `key` and `value` to be 3-D but found 2-D and 3-D tensors respectively"
        # 3D query, 2D key and 3D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, torch.randn(3, 3), value, need_weights=False)

        # Currently, this case will just go to the slow path and get
        # the usual message because it fails the requirement to be
        # batched.
        msg = "expected `key` and `value` to be 3-D but found 3-D and 2-D tensors respectively"
        # 3D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, torch.randn(3, 3), need_weights=False)

        msg = "expected `key_padding_mask` to be `None` or 2-D but found 1-D tensor instead"
        # 3D query, 3D key, 3D value and 1D key_padding_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                key_padding_mask=torch.tensor([False, True, True], dtype=torch.bool),
                need_weights=False,
            )

        msg = (
            "expected `attn_mask` to be `None`, 2-D or 3-D but found 1-D tensor instead"
        )
        # 3D query, 3D key, 3D value and 1D attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                attn_mask=torch.tensor([False, True, True], dtype=torch.bool),
                need_weights=False,
            )

        # Unbatched (2D) query cases
        # NOTE: error messages are the same as regular path because the fast path doesn't support 2D.
        query = torch.randn(4, 4)
        key = torch.randn(4, 4)
        value = torch.randn(4, 4)

        msg = "expected `key` and `value` to be 2-D but found 3-D and 2-D tensors respectively"
        # 2D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, torch.randn(4, 4, 4), value)

        msg = "expected `key` and `value` to be 2-D but found 2-D and 3-D tensors respectively"
        # 2D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, torch.randn(4, 4, 4))

        msg = "expected `key_padding_mask` to be `None` or 1-D but found 2-D tensor instead"
        # 2D query, 2D key, 2D value and 1D key_padding_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                key_padding_mask=torch.tensor(
                    [[False, False, True, True] * 2], dtype=torch.bool
                ),
            )

        msg = (
            "expected `attn_mask` to be `None`, 2-D or 3-D but found 1-D tensor instead"
        )
        # 2D query, 2D key, 2D value and 1D attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                attn_mask=torch.tensor([False, False, True, True], dtype=torch.bool),
            )

        msg = r"Expected `attn_mask` shape to be \(4, 4, 4\)"
        # 2D query, 2D key, 2D value and 3D incorrect attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                attn_mask=torch.randn(5, 4, 4).bernoulli_().to(torch.bool),
            )

    def test_multihead_attn_nested_tensor_outside_fast_path(self):
        # [FUNCTION DOC] Checks NestedTensor handling. With grad enabled and parameters
        # requiring grad the module must raise AssertionError (message depends on whether a
        # torch function override is present, in which case the test returns early). Works
        # under no_grad and inference mode, and also once the nested tensor and all projection
        # parameters have requires_grad turned off. Mutates the module parameters
        # requires_grad flags.
        mha = torch.nn.MultiheadAttention(4, 4, batch_first=True).eval()
        nt = torch.nested.nested_tensor([torch.randn(4, 4)])
        # One tested platform (linux-bionic-py3.7-clang) has a torch_function for one
        # or more of these. Take advantage of that to test the torch_function bailout.
        has_torch_func = torch.overrides.has_torch_function(
            (
                nt,
                mha.in_proj_weight,
                mha.in_proj_bias,
                mha.out_proj.weight,
                mha.out_proj.bias,
            )
        )
        if has_torch_func:
            msg = "MultiheadAttention does not support NestedTensor.*argument has_torch_function"
        else:
            msg = (
                "MultiheadAttention does not support NestedTensor outside of its fast path.*grad is "
                + "enabled and.*or biases requires_grad"
            )
        with self.assertRaisesRegex(AssertionError, msg):
            mha(nt, nt, nt)

        if has_torch_func:
            # Just give up, they're all going to fail with the same message.
            return

        with torch.no_grad():
            mha(nt, nt, nt)
        with torch.inference_mode():
            mha(nt, nt, nt)
        nt = torch.nested.nested_tensor([torch.randn(4, 4, requires_grad=False)])
        nt.requires_grad = False
        with self.assertRaisesRegex(AssertionError, msg):
            mha(nt, nt, nt)
        mha.in_proj_weight.requires_grad = False
        mha.in_proj_bias.requires_grad = False
        mha.out_proj.weight.requires_grad = False
        mha.out_proj.bias.requires_grad = False
        mha(nt, nt, nt)


class TestMultiheadAttentionNNDeviceType(NNTestCase):
    # [CLASS DOC] Device-generic tests for MultiheadAttention, instantiated per device by the
    # device-type test machinery. Has no state of its own; each test takes device (and
    # sometimes dtype) arguments.
    def test_multihead_self_attn_two_masks_fast_path(self, device):
        # [FUNCTION DOC] Under no_grad, runs a batch_first eval module with both an attention
        # mask and a key padding mask (fast path) and compares to the functional slow path
        # implementation. Rows completely masked out are nan, so they are zeroed in both
        # results before comparing with assertEqual.
        """
        Multihead self-attention should give the same result on the fast path (BetterTransformer) as on the slow path
        when both attention mask (mask type 0) and key padding mask (mask type 1) are provided
        """
        with torch.no_grad():
            embed_dim = 14
            num_heads = 7
            batch_size = 8
            src_len = 5

            query = value = key = torch.rand(batch_size, src_len, embed_dim).to(device)
            # Create masks of two different types
            attn_mask = torch.randint(0, 2, (src_len, src_len)).bool().to(device)
            key_padding_mask = (
                torch.randint(0, 2, (batch_size, src_len)).bool().to(device)
            )

            # We'll need expanded versions of the masks for masking out the outputs below
            attn_mask_expanded = attn_mask.reshape(1, 1, src_len, src_len).expand(
                batch_size, num_heads, src_len, src_len
            )
            key_padding_mask_expanded = key_padding_mask.reshape(
                batch_size, 1, 1, src_len
            ).expand(batch_size, num_heads, src_len, src_len)
            merged_mask = attn_mask_expanded.logical_or(key_padding_mask_expanded)

            # Compute attention on the fast path
            mta_model = torch.nn.MultiheadAttention(
                embed_dim, num_heads, batch_first=True, device=device
            )
            mta_model.training = False
            result_fast_path, _ = mta_model(
                query,
                key,
                value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

            # Compute attention on the slow path
            result_ref, _ = torch.nn.functional.multi_head_attention_forward(
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
                embed_dim,
                num_heads,
                mta_model.in_proj_weight,
                mta_model.in_proj_bias,
                mta_model.bias_k,
                mta_model.bias_v,
                mta_model.add_zero_attn,
                mta_model.dropout,
                mta_model.out_proj.weight,
                mta_model.out_proj.bias,
                training=mta_model.training,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
                use_separate_proj_weight=False,
                q_proj_weight=mta_model.q_proj_weight,
                k_proj_weight=mta_model.k_proj_weight,
                v_proj_weight=mta_model.v_proj_weight,
                average_attn_weights=False,
            )
            result_ref = result_ref.transpose(0, 1)  # Convert to batch-first

            # Rows which are completely masked out are nan, we need to exclude them from comparison
            mask_out = (
                merged_mask[:, 0, :, :]
                .all(-1, keepdim=True)
                .expand(batch_size, src_len, embed_dim)
            )
            result_fast_path_masked = result_fast_path.masked_fill(mask_out, 0)
            result_ref_masked = result_ref.masked_fill(mask_out, 0)

            self.assertEqual(result_fast_path_masked, result_ref_masked)

    @torch.no_grad()
    @unittest.skipIf(
        TEST_WITH_CROSSREF,
        "CrossRef turns on TorchFunctionMode, and so disables fastpath.",
    )
    def test_multihead_self_attn_two_masks_fast_path_mock(self, device):
        # [FUNCTION DOC] Patches the native multi head attention op with a mock and runs an
        # eval batch_first module with both masks, then asserts the mock was called, proving
        # the fast path was taken. Skips on devices other than cpu, cuda, xpu or the
        # private-use backend; skipped under CrossRef. Autocast is disabled during the call.
        """
        Multihead self-attention should take fast path when both attention mask (mask type 0)
        and key padding mask (mask type 1) are provided at the same time on CPU and CUDA and PrivateUse1
        """
        device = device.rstrip(":0123456789")
        if device not in [
            "cpu",
            "cuda",
            "xpu",
            torch._C._get_privateuse1_backend_name(),
        ]:
            self.skipTest("Fastpath only runs on CPU and CUDA and XPU and PrivateUse1.")

        with torch.autocast(device_type=device, enabled=False):
            embed_dim = 16
            num_heads = 8
            batch_size = 8
            src_len = 5

            query = value = key = torch.rand(batch_size, src_len, embed_dim).to(device)
            # Create masks of two different types
            attn_mask = torch.randint(0, 2, (src_len, src_len)).bool().to(device)
            key_padding_mask = (
                torch.randint(0, 2, (batch_size, src_len)).bool().to(device)
            )

            with mock.patch(
                "torch._native_multi_head_attention",
                new=mock.MagicMock(return_value=(torch.Tensor(), torch.Tensor())),
            ) as fastpath_mock:
                # Compute attention on the fast path
                mta_model = torch.nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, device=device
                ).eval()
                mta_model.training = False
                mta_model(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
                # If mock was called, fastpath was taken
                self.assertTrue(fastpath_mock.called)

    @onlyOn(["cuda", "xpu", torch._C._get_privateuse1_backend_name()])
    @dtypes(torch.half, torch.float, torch.double)
    def test_multihead_attention_dtype(self, device, dtype):
        # [FUNCTION DOC] Runs a default (sequence-first) module cast to the given dtype and
        # moved to the device, on random q, k, v, and asserts the output keeps the query size
        # and dtype. Runs only on cuda, xpu or private-use devices for half, float and double.
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        model = nn.MultiheadAttention(embed_dim, num_heads).to(device).to(dtype)
        q = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        k = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        v = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        out = model(q, k, v)
        self.assertEqual(q.size(), out[0].size())
        self.assertEqual(dtype, out[0].dtype)

    @onlyOn(["cuda", "xpu", torch._C._get_privateuse1_backend_name()])
    @dtypes(torch.half, torch.float, torch.double)
    def test_multihead_attention_dtype_batch_first(self, device, dtype):
        # [FUNCTION DOC] Same dtype and size check for a batch_first module, run once in
        # training mode and once in eval mode under no_grad (fast path eligible, need_weights
        # disabled).
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        # With batch_first=True, we have the possibility of hitting
        # the native fast path if we call .eval() and enable inference
        # mode. Test both paths.
        for training in (True, False):
            model = (
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                .to(device)
                .to(dtype)
            )
            if not training:
                model = model.eval()
                cm = torch.no_grad()
            else:
                cm = contextlib.nullcontext()
            with cm:
                q = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype)
                k = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype)
                v = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype)
                # fast path currently doesn't support weights
                out = model(q, k, v, need_weights=False)
                self.assertEqual(q.size(), out[0].size())
                self.assertEqual(dtype, out[0].dtype)

    @dtypes(torch.double)
    @torch.no_grad()
    def test_multihead_attn_fast_path_query_and_bias_have_different_dtypes(
        self, device, dtype
    ):
        # [FUNCTION DOC] Smoke test (no assertions beyond not raising): replaces the input
        # projection bias with a half precision parameter while query is double, then runs the
        # eval batch_first module under no_grad.
        mha = torch.nn.MultiheadAttention(
            4, 4, batch_first=True, dtype=dtype, device=device
        ).eval()
        mha.in_proj_bias = torch.nn.Parameter(
            mha.in_proj_bias.to(torch.half).to(device)
        )
        query = torch.randn(4, 4, 4, dtype=dtype, device=device)
        mha(query, query, query)

    @dtypes(torch.double)
    @torch.no_grad()
    def test_multihead_attn_fast_path_small_test(self, device, dtype):
        # [FUNCTION DOC] Smoke test: runs a small eval batch_first double module under no_grad
        # on random input; passes if no exception is raised.
        mha = torch.nn.MultiheadAttention(
            4, 4, batch_first=True, dtype=dtype, device=device
        ).eval()
        query = torch.randn(4, 4, 4, dtype=dtype, device=device)
        mha(query, query, query)

    @dtypes(torch.double)
    @xfailIfNoAcceleratorTriton
    def test_fast_path_check_with_mask_does_not_break_in_compile(self, device, dtype):
        # [FUNCTION DOC] Compiles an eval TransformerEncoder with fullgraph and calls it with
        # a source key padding mask, to ensure the fast path mask check (left alignment) does
        # not break torch.compile. Passes if no exception is raised; marked xfail when no
        # accelerator Triton is available. Related to issue 163640.
        # Test TransformerEncoder fast path determination with src_key_padding_mask set.
        # Specifically, ensure the mask left-align check doesn't fail in torch.compile.
        # See https://github.com/pytorch/pytorch/issues/163640
        layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            batch_first=True,
            dropout=0.1,
            device=device,
            dtype=dtype,
        )
        encoder = nn.TransformerEncoder(layer, num_layers=2).eval()
        encoder = torch.compile(encoder, fullgraph=True)
        x = torch.randn(1, 41, 512, dtype=dtype, device=device)
        pad_mask = torch.rand(1, 41, device=device) > 0.5
        pad_mask[..., 0] = True
        encoder(x, mask=None, src_key_padding_mask=pad_mask)

    @dtypes(torch.double)
    @torch.no_grad()
    def test_multihead_attn_in_proj_bias_none(self, device, dtype):
        # [FUNCTION DOC] Smoke test: module built without bias must run forward under no_grad
        # (in_proj_bias is None).
        mha = torch.nn.MultiheadAttention(2, 2, bias=False, dtype=dtype, device=device)
        query = torch.rand(2, 2, 2, dtype=dtype, device=device)
        mha(query, query, query)

    @dtypes(torch.double)
    @torch.no_grad()
    def test_multihead_attn_in_proj_weight_none(self, device, dtype):
        # [FUNCTION DOC] Smoke test: with key and value dims different from the embedding dim
        # the module uses separate projection weights and the packed weight is None; forward
        # must still run under no_grad.
        # Setting kdim == vdim == 2 means that vdim != embed_dim
        # will cause the logic to use per-input project weights, thereby
        # forcing self.in_proj_weight = None
        mha = torch.nn.MultiheadAttention(
            4, 4, vdim=2, kdim=2, dtype=dtype, device=device
        )
        query = torch.rand(4, 4, 4, dtype=dtype, device=device)
        key = torch.rand(4, 4, 2, dtype=dtype, device=device)
        mha(query, key, key)


instantiate_device_type_tests(TestMultiheadAttentionNNDeviceType, globals())
instantiate_parametrized_tests(TestMultiheadAttentionNN)

if __name__ == "__main__":
    run_tests()
