# [MODULE DOC] Example of the first-class dimensions API from functorch.dim: a BERT style self
# attention module written with explicit dimension objects instead of positional reshapes,
# using a linear layer variant also built from dimension objects. Counterpart of
# attn_positional.py, used by tests and benchmarks to compare the two styles. No import-time
# side effects beyond imports.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
from functorch.dim import cat, dimlists, dims
from torch import nn


class Linear(nn.Linear):
    # [CLASS DOC] Linear layer subclass whose forward is expressed with first-class
    # dimensions: it binds dimension objects to the input and the weight, multiplies, and
    # reduces over the input feature dimension. Parameters are inherited unchanged from the
    # standard linear module.
    def forward(self, input):
        # [FUNCTION DOC] Computes the linear transformation with first-class dimensions.
        # Creates a feature-in and feature-out dimension and a list of batch dimensions,
        # indexes the input by batch and input features, indexes the weight by output and
        # input features, multiplies, sums over the input features, adds the bias indexed by
        # output features, then reorders the result back to a regular tensor with batch
        # dimensions followed by output features. Returns that tensor; accepts any number of
        # leading batch dimensions.
        ci, co = dims()
        b = dimlists()
        result = (input[b, ci] * self.weight[co, ci]).sum(ci) + self.bias[co]
        return result.order(b, co)


class BertSelfAttention(nn.Module):
    # [CLASS DOC] BERT style self attention implemented with first-class dimensions. Holds
    # query, key and value projections built by the supplied linear factory, the dropout
    # probability, optional relative position embedding type and, when that type is given, a
    # distance embedding table. Invariant: hidden size is divisible by the head count.
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        position_embedding_type=None,
        max_position_embeddings=None,
        linear=Linear,
    ):
        # [FUNCTION DOC] Constructor. Arguments: hidden_size, num_attention_heads,
        # attention_probs_dropout_prob, optional position_embedding_type (relative_key or
        # relative_key_query style) with max_position_embeddings, and a linear factory
        # defaulting to the dimension based Linear. Raises ValueError if hidden size is not a
        # multiple of the head count, and AssertionError if a position embedding type is set
        # without a max length. Creates the three projections and, for relative positions, an
        # embedding table of twice the max length minus one rows.
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = linear(hidden_size, self.all_head_size)
        self.key = linear(hidden_size, self.all_head_size)
        self.value = linear(hidden_size, self.all_head_size)

        self.dropout_prob = attention_probs_dropout_prob
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

    def forward(
        self,
        hidden_states,
        past_key_value=None,
    ):
        # [FUNCTION DOC] Runs attention on hidden states, with optional cached key/value pair
        # past_key_value. Projects q, k, v, then binds them to named dimensions, splitting the
        # hidden axis into heads and features. If a past is given, concatenates cached keys
        # and values along the key sequence into a longer dimension. Scores are the scaled dot
        # product over features. For relative position types adds scores derived from an
        # embedding lookup indexed by query minus key distance, raising AssertionError if the
        # key length exceeds the maximum positions. Applies softmax over the key sequence,
        # dropout (applied regardless of training mode, since the training flag is not
        # passed), weights the values, and returns the context reordered to batch, query
        # sequence, merged heads and features.
        # first run the encoding linear layers for q, k, v normally
        # the meaning of a linear layer is well understood, so no need to use explicit dimensions
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # introduce values that represent each dimension. dimensions are 'first class'
        # because they are actual python values introduced here
        batch, query_sequence, key_sequence, heads, features = dims()
        heads.size = self.num_attention_heads

        # bind the positional dimensions in k, q, and v against
        # our values. the sizes of each dimension are determined by this binding
        # and when a dimension is used twice (e.g. batch), its size against both
        # uses is checked for consistency.
        # The group (heads, features) splits apart a single positional dimension
        # into two dimensions. Since heads.size*features.size == q.size(2)
        # and we specified heads.size, features.size is inferred here.
        q = q[batch, query_sequence, [heads, features]]
        k = k[batch, key_sequence, [heads, features]]
        v = v[batch, key_sequence, [heads, features]]

        # this option allows the model to attend to not just the elements of the current sequence
        # but the previous elements as well as additional tokens.
        if past_key_value is not None:
            extended_key_sequence = dims()
            key_past = past_key_value[0][batch, heads, key_sequence, features]
            value_past = past_key_value[1][batch, heads, key_sequence, features]
            # cat introduces a new dimension extended_key_sequence, because it is twice as long
            # as the original key_sequence
            k = cat([key_past, k], key_sequence, extended_key_sequence)
            v = cat([value_past, v], key_sequence, extended_key_sequence)
            # for the rest of the function, we will just use extended_key_sequence in lieu of
            # key_sequence
            key_sequence = extended_key_sequence

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # The actual outer-product and summation are explicitly represented here,
        # and like einsum, will be pattern matched to an efficient matrix multiply op.
        attention_scores = (q * k).sum(features) / math.sqrt(features.size)

        # relative positional embeddings gave a unique embedding based on the distance between
        # key and value tokens in the sequence, e.g.
        #  0  1  2  3
        # -1  0  1  2
        # -2 -1  0  1
        # -3 -2 -1  0
        if self.position_embedding_type is not None:
            # the value of a dimension object when used as a tensor is the indices along its dimension
            # so we can directly subtract the two dimensions to get a 2D tensor of (query_sequence x key_sequence)
            # with the distance between them
            distance = query_sequence - key_sequence

            if not (key_sequence.size <= self.max_position_embeddings):
                raise AssertionError(
                    f"key_sequence.size ({key_sequence.size}) exceeds "
                    f"max_position_embeddings ({self.max_position_embeddings})"
                )

            # we can then use that as an indirect index into the embedding table values to look up the features for that index
            # this is just a `gather` primitive op. The resulting tensor will
            # have all the dimensions of embedding_idx (query_sequence x key_sequence),
            # plus all the dimensions of `embed` that were not indirectly accessed (`embedding_range`).
            # this form of indirect indexing is more straightforward than either advanced indexing or torch.gather which both
            # have a lot of dependencies on the positions of indexing tensors.

            positional_embedding = self.distance_embedding.weight[
                self.max_position_embeddings - 1 + distance, features
            ]

            if self.position_embedding_type == "relative_key":
                # these were einsum ops in the positional code because they are not easy to fit to existing matmul operators
                # even though they are degenerate matmuls
                relative_position_scores = (q * positional_embedding).sum(features)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = (q * positional_embedding).sum(
                    features
                )
                relative_position_scores_key = (k * positional_embedding).sum(features)
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_probs = attention_scores
        # Normalize the attention scores to probabilities.
        attention_probs = torch.softmax(attention_scores, dim=key_sequence)
        # # This is actually dropping out entire tokens to attend to, which might
        # # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = torch.nn.functional.dropout(
            attention_probs, p=self.dropout_prob
        )

        # similarly, we can replace the matmul with a direct listing of the outer product, which makes it clear
        # we are weighting the values v across all keys with the attention scores.
        context_layer = (attention_probs * v).sum(key_sequence)

        # finally, we convert back to a standard tensor by describing the layout of dimensions.
        # working in reverse to with_dims, the (heads, features) group flattens the dimensions into a single one.
        return context_layer.order(batch, query_sequence, [heads, features])
