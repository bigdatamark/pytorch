# [MODULE DOC] Reference BERT style self attention written with ordinary positional tensor
# operations (view, permute, matmul, einsum). Counterpart of attn_ft.py, which expresses the
# same computation with first-class dimensions. No import-time side effects beyond imports.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn


class BertSelfAttention(nn.Module):
    # [CLASS DOC] BERT style multi-head self attention module. State: query, key and value
    # linear projections, a dropout layer, head count and head size, and, when a relative
    # position embedding type is configured, a maximum position count and a distance embedding
    # table. Invariant: hidden size is divisible by the head count.
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        position_embedding_type=None,
        max_position_embeddings=None,
    ):
        # [FUNCTION DOC] Constructor. Arguments: hidden_size, num_attention_heads, dropout
        # probability, optional position_embedding_type with max_position_embeddings. Raises
        # ValueError when hidden size is not a multiple of the head count, and AssertionError
        # when a position embedding type is given without a maximum position count. Builds the
        # projections, dropout module and, for relative positions, an embedding of twice the
        # maximum minus one rows.
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type

        if self.position_embedding_type is not None:
            if max_position_embeddings is None:
                raise AssertionError(
                    "max_position_embeddings must not be None when "
                    f"position_embedding_type is {self.position_embedding_type}"
                )
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * max_position_embeddings - 1, self.attention_head_size
            )

    def transpose_for_scores(self, x):
        # [FUNCTION DOC] Reshapes a tensor of batch, sequence, all-heads features into batch,
        # heads, sequence, head size by splitting the last axis into heads and head size and
        # moving the heads axis ahead of the sequence axis. Returns a permuted view.
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        past_key_value=None,
    ):
        # [FUNCTION DOC] Computes self attention for hidden states, with an optional cached
        # key/value pair past_key_value whose tensors are prepended along the sequence axis.
        # Scores are the scaled dot product of queries and keys; for the relative key types an
        # embedding of the query/key position distance (cast to the query dtype) contributes
        # extra einsum terms. Applies softmax over the last axis, dropout (active only in
        # training mode), weights the values and merges heads back into a batch by sequence by
        # hidden tensor which is returned.
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if self.position_embedding_type is not None:
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=q.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", q, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", q, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", k, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_probs = attention_scores
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # # This is actually dropping out entire tokens to attend to, which might
        # # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, v)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
