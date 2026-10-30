# flake8: noqa: E266, C417
# [MODULE DOC] Compact Mixtral-style mixture-of-experts decoder-only transformer used by the
# gpt-fast benchmarks (adapted from the gpt-fast project). Contains model arguments with a
# registry of named configurations (only Mixtral 8x7B with 16 layers), a key-value cache,
# the transformer with grouped query attention and rotary embeddings, a top-k routed expert
# feed-forward block, RMS normalization, and rotary helper functions. Weights are left
# uninitialized or default-initialized; the benchmark overwrites them with random values. No
# import-time side effects besides defining the configuration mapping.
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    # [FUNCTION DOC] HELPER: Round n up to the nearest multiple of k and return it (n itself if
    # it already divides evenly).
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    # [CLASS DOC] Dataclass of model hyperparameters: context length, vocabulary size, layer
    # count, attention head count, model width, feed-forward width (derived when none), number
    # of key-value heads, per-head width, rotary base, norm epsilon, number of experts and
    # number of experts activated per token. Post-initialization fills in derived fields.
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    num_experts: int = 8
    num_activated_experts: int = 2

    def __post_init__(self):
        # [FUNCTION DOC] HELPER: Fill derived fields after construction: the number of key-value
        # heads defaults to the attention head count when it is minus one; the feed-forward
        # width defaults to two thirds of four times the model width rounded up to a multiple of
        # 256; and the per-head width is always recomputed as model width divided by head count,
        # overriding any value passed in.
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        # [FUNCTION DOC] Build model arguments from a configuration name. An exact key in the
        # configuration mapping is used directly. Otherwise a fuzzy search collects keys that
        # occur in the upper-cased or the original name; exactly one match is required, else an
        # AssertionError is raised stating the count. Returns a new ModelArgs.
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [
            config
            for config in transformer_configs
            if config in str(name).upper() or config in str(name)
        ]
        if len(config) != 1:
            raise AssertionError(
                f"Expected exactly one config match for '{name}', but got {len(config)}: {config}"
            )
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "Mixtral-8x7B-v0.1": dict(
        block_size=32768,
        n_layer=16,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        rope_base=1000000.0,
        num_experts=8,
        num_activated_experts=2,
    ),
}


class KVCache(nn.Module):
    # [CLASS DOC] Preallocated key and value cache for one attention layer. Holds two
    # zero-initialized buffers named k_cache and v_cache of shape batch, heads, sequence length,
    # head width, in the given dtype (bfloat16 by default). Update writes new key and value
    # slices in place at the given positions.
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        # [FUNCTION DOC] HELPER: Allocate the zeroed key and value cache buffers for the given
        # batch size, maximum sequence length, number of heads and head width.
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # [FUNCTION DOC] Write new keys and values into the cache at the given positions along
        # the sequence dimension and return the whole key and value buffers. Raises an
        # AssertionError if the number of positions differs from the sequence length of the new
        # keys. Mutates the buffers in place.
        # input_pos: [S], k_val: [B, H, S, D]
        if input_pos.shape[0] != k_val.shape[2]:
            raise AssertionError(
                f"input_pos.shape[0] ({input_pos.shape[0]}) must equal k_val.shape[2] ({k_val.shape[2]})"
            )

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Transformer(nn.Module):
    # [CLASS DOC] The decoder-only language model: a token embedding, a list of transformer
    # blocks, a final RMS norm and a bias-free output projection to vocabulary logits. Keeps
    # rotary frequencies, a causal mask and the maximum batch size and sequence length of the
    # allocated caches, all initialized lazily by setup_caches. Forward returns logits for the
    # given tokens at the given positions.
    def __init__(self, config: ModelArgs) -> None:
        # [FUNCTION DOC] HELPER: Create the embedding, the stack of blocks sized by the layer
        # count, the final norm and the output layer, and initialize cache bookkeeping (rotary
        # frequencies and mask cache none, maximum batch size and sequence length minus one).
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Tensor | None = None
        self.mask_cache: Tensor | None = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        # [FUNCTION DOC] Allocate inference state. Returns immediately if existing caches
        # already cover the requested batch size and sequence length. Otherwise it rounds the
        # sequence length up to a multiple of eight, records the sizes, installs a fresh
        # key-value cache in the attention of every block, computes the rotary frequency table
        # for the full block size, and builds a lower-triangular boolean causal mask.
        # KNOWN ISSUE: the constructor declares a mask cache attribute that is never used, while
        # the actual mask is stored in a different attribute created only here; the rotary table
        # and the mask are created on the default device, not the device of the model.
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_local_heads, head_dim
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
        )
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )

    def forward(self, idx: Tensor, input_pos: Tensor | None = None) -> Tensor:
        # [FUNCTION DOC] Compute logits for a batch of token ids. Looks up the causal mask rows
        # and rotary frequencies for the given positions, embeds the tokens, runs every block,
        # applies the final norm and the output projection. Raises an AssertionError if the
        # caches were not set up. Passing no positions fails when indexing.
        if self.freqs_cis is None:
            raise AssertionError("Caches must be initialized first")
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        # [FUNCTION DOC] Construct the model from a named configuration through the model
        # arguments lookup.
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    # [CLASS DOC] One decoder layer: attention with a pre-norm and a residual connection,
    # followed by a mixture-of-experts feed-forward with its own pre-norm and residual
    # connection. Holds the attention, expert feed-forward and two RMS norms as submodules.
    def __init__(self, config: ModelArgs) -> None:
        # [FUNCTION DOC] HELPER: Create the attention, the expert feed-forward and the two RMS
        # norm modules.
        super().__init__()
        self.attention = Attention(config)
        self.block_sparse_moe = MOEFeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        # [FUNCTION DOC] Apply attention to the normalized input and add the residual, then
        # apply the expert feed-forward to the normalized result and add the residual again.
        # Returns the layer output.
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.block_sparse_moe(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    # [CLASS DOC] Grouped query self-attention with a fused query, key and value projection,
    # rotary position embeddings and an optional key-value cache that is attached later by the
    # transformer. Keeps the head counts, per-head width and model width. A state dict pre-load
    # hook converts checkpoints with separate query, key and value weights into the
    # fused layout.
    def __init__(self, config: ModelArgs):
        # [FUNCTION DOC] HELPER: Create the fused projection and output projection (both without
        # bias), record the head sizes, set the cache to none and register the checkpoint
        # conversion hook. Raises an AssertionError if the model width is not divisible by the
        # head count.
        super().__init__()
        if config.dim % config.n_head != 0:
            raise AssertionError(
                f"config.dim ({config.dim}) must be divisible by config.n_head ({config.n_head})"
            )

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        # [FUNCTION DOC] HELPER: Pre-load state dict hook. If the state dict has separate query
        # weights under the module prefix, removes the separate query, key and value weights and
        # stores their concatenation along the first dimension under the fused weight name.
        # Mutates the given state dict in place; extra positional arguments are ignored.
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Tensor | None = None,
    ) -> Tensor:
        # [FUNCTION DOC] Run self-attention on a batch of shape batch, sequence, width. Splits
        # the fused projection into queries, keys and values, reshapes them into heads, applies
        # rotary embeddings to queries and keys, moves heads before the sequence axis,
        # optionally writes into and reads from the key-value cache at the given positions,
        # repeats the key and value heads to match the query heads, runs scaled dot product
        # attention with the boolean mask and no dropout, merges the heads, and applies the
        # output projection. Returns a tensor of the input shape.
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class ConditionalFeedForward(nn.Module):
    # [CLASS DOC] Expert weights for a gated feed-forward over several experts. Holds three
    # parameters stacked over experts: the up projection, the down projection and the gate
    # projection (named first, second and third weight), created uninitialized. Forward gathers
    # the weights of the selected experts per token and applies them.
    def __init__(self, config):
        # [FUNCTION DOC] HELPER: Allocate the three uninitialized expert weight stacks with
        # shapes derived from the number of experts, feed-forward width and model width.
        super().__init__()
        self.w1 = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size, config.dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.intermediate_size)
        )
        self.w3 = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size, config.dim)
        )

    def forward(self, x: Tensor, expert_indices: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply the selected experts to each token. Inputs are token features of
        # shape tokens by width and expert indices of shape tokens by activated experts. The
        # weights of the chosen experts are gathered for every token, then a gated activation is
        # computed (silu of the first projection times the third projection) and the second
        # projection brings it back to model width. Returns a tensor of shape tokens, activated
        # experts, width, one output per selected expert.
        w1_weights = self.w1[expert_indices]  # [T, A, D, D]
        w3_weights = self.w3[expert_indices]  # [T, A, D, D]
        w2_weights = self.w2[expert_indices]  # [T, A, D, D]
        x1 = F.silu(torch.einsum("ti,taoi -> tao", x, w1_weights))
        x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
        expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
        return expert_outs


class MOEFeedForward(nn.Module):
    # [CLASS DOC] Mixture-of-experts feed-forward block: a bias-free linear router scoring all
    # experts and a conditional feed-forward holding the expert weights. Stores the model width
    # and the number of experts activated per token.
    def __init__(self, config) -> None:
        # [FUNCTION DOC] HELPER: Create the router linear layer, the conditional feed-forward
        # and remember the width and activated expert count.
        super().__init__()
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(config)
        self.dim = config.dim
        self.num_activated_experts = config.num_activated_experts

    def forward(self, x: Tensor) -> Tensor:
        # [FUNCTION DOC] Route each token to its top experts and combine their outputs. The
        # input is flattened to rows of model width, router scores are turned into probabilities
        # with softmax, the top activated experts are chosen per token and their weights are
        # renormalized to sum to one (in place), the chosen experts are evaluated and the
        # outputs are summed with those weights. Returns a two-dimensional tensor of tokens
        # by width.
        # KNOWN ISSUE: the output is not reshaped back to batch and sequence dimensions, so the
        # residual addition in the transformer block only works for a batch of one
        # through broadcasting.
        x = x.view(-1, self.dim)
        # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
        # x: [T, D]
        scores = self.gate(x)  # [T, E]
        expert_weights = F.softmax(scores, dim=-1)
        expert_weights, expert_indices = torch.topk(
            expert_weights, self.num_activated_experts, dim=-1
        )  # [T, A], [T, A]
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)  # [T, A]
        expert_outs = self.cond_ffn(x, expert_indices)
        return torch.einsum("tai,ta -> ti", expert_outs, expert_weights)


class RMSNorm(nn.Module):
    # [CLASS DOC] Root mean square layer normalization with a learnable per-feature scale
    # initialized to ones and a small epsilon for numerical stability.
    def __init__(self, dim: int, eps: float = 1e-5):
        # [FUNCTION DOC] HELPER: Store the epsilon and create the learnable scale vector of ones
        # with the given width.
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # [FUNCTION DOC] HELPER: Scale the input by the reciprocal square root of the mean of
        # squares over the last dimension plus epsilon.
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        # [FUNCTION DOC] Normalize the input in float32, cast the result back to the input dtype
        # and multiply by the learnable scale. Returns a tensor of the input shape.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    # [FUNCTION DOC] Precompute rotary embedding tables. Inverse frequencies are derived from
    # the base for each pair of channels, outer-multiplied with positions up to the sequence
    # length, converted to unit-magnitude complex numbers, and stored as stacked real and
    # imaginary parts in the last dimension. Returns a bfloat16 tensor of shape sequence length,
    # half the element count, two.
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    # [FUNCTION DOC] Apply rotary position embedding to a tensor of shape batch, sequence,
    # heads, head width. The tensor is converted to float32 and its last dimension viewed as
    # pairs; the frequency table is reshaped to broadcast over batch and heads; each pair is
    # rotated by the matching angle, which is a complex multiplication written with real
    # arithmetic. Returns a tensor of the input shape cast back to the input dtype.
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
