# flake8: noqa: E266, C417
# [MODULE DOC] Compact Llama-style decoder-only transformer used by the gpt-fast benchmarks.
# Defines a configuration dataclass with named presets, a KV cache, the transformer with its block,
# attention (grouped-query, rotary embeddings), feed-forward and RMS norm layers, plus rotary-
# embedding helpers. No import-time side effects beyond building the preset table.
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    # [FUNCTION DOC] Rounds n up to the nearest multiple of k and returns it (n itself if already
    # one).
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    # [CLASS DOC] Dataclass of model hyperparameters (block size, vocab size, layers, heads, width,
    # feed-forward size, number of key/value heads, head dim, rope base, norm epsilon). Defaults
    # describe a 7B Llama. After construction, derived fields are filled in by the post-init step.
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

    def __post_init__(self):
        # [FUNCTION DOC] Fills derived fields: key/value head count defaults to the head count when
        # negative, the feed-forward size defaults to two thirds of four times the width rounded up
        # to a multiple of 256 when unset, and head_dim is always recomputed as width divided by
        # head count (overriding any value passed in).
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        # [FUNCTION DOC] Builds a ModelArgs from a preset name. An exact key in the preset table is
        # used directly; otherwise presets whose key occurs in the upper-cased or original name are
        # candidates and the longest key wins. Raises AssertionError if the two longest candidates
        # have equal length.
        # KNOWN ISSUE: when no preset matches, indexing the empty candidate list raises IndexError
        #   rather than a descriptive error.
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [
            config
            for config in transformer_configs
            if config in str(name).upper() or config in str(name)
        ]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            if len(config[0]) == len(config[1]):
                raise AssertionError(
                    f"Ambiguous config match for '{name}': {config[0]} and {config[1]} have same length"
                )  # make sure only one 'best' match

        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(
        block_size=16384, vocab_size=32000, n_layer=32, dim=4096, rope_base=1000000
    ),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(
        n_layer=48,
        n_head=64,
        dim=8192,
        vocab_size=32000,
        n_local_heads=8,
        intermediate_size=22016,
        rope_base=1000000,
    ),  # CodeLlama-34B-Python-hf
    "70B": dict(
        n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672
    ),
    "Mistral-7B": dict(
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=32000,
    ),
}


class KVCache(nn.Module):
    # [CLASS DOC] Static key/value cache for attention. Holds two zero-initialized buffers, one for
    # keys and one for values, each shaped batch, heads, max sequence length, head dim, in the
    # given dtype (bfloat16 by default). Updated in place at given positions.
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        # [FUNCTION DOC] Allocates the zeroed key and value cache buffers from the given sizes and
        # dtype.
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # [FUNCTION DOC] Writes new keys and values into the cache at the sequence positions listed
        # in input_pos (in place along the sequence dimension) and returns the whole key and value
        # cache tensors. Raises AssertionError when the number of positions differs from the
        # sequence length of the new keys.
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
    # [CLASS DOC] Decoder-only transformer: token embedding, a list of transformer blocks, a final
    # RMS norm and a bias-free output projection to vocabulary logits. Caches (KV caches, rotary
    # table, causal mask) are created lazily by setup_caches and must exist before forward.
    def __init__(self, config: ModelArgs) -> None:
        # [FUNCTION DOC] Builds the submodules from the config and initializes the lazily created
        # state: rotary table and mask cache are None and the batch and sequence capacities are
        # minus one.
        # KNOWN ISSUE: mask_cache is never used; the mask is stored later under the name
        #   causal_mask, which is not declared here.
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
        # [FUNCTION DOC] Prepares inference state for the given maximum batch and sequence length.
        # Returns immediately if existing capacity already covers both. Otherwise rounds the
        # sequence length up to a multiple of 8, records both capacities, gives each block's
        # attention a fresh KVCache, computes the rotary table for the full block size, and builds
        # a lower-triangular boolean causal mask of the rounded sequence length.
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
        # [FUNCTION DOC] Computes logits for token ids idx at the cache positions input_pos. Raises
        # AssertionError if setup_caches has not been called. Selects the causal mask rows and
        # rotary entries for the positions, embeds tokens, runs all blocks, applies the final norm
        # and output projection, and returns the logits tensor.
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
        # [FUNCTION DOC] Alternate constructor: looks up a config by preset name and builds the
        # model.
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    # [CLASS DOC] One pre-norm transformer layer: attention and feed-forward sub-layers, each
    # applied to a normalized input and added back as a residual.
    def __init__(self, config: ModelArgs) -> None:
        # [FUNCTION DOC] Creates the attention, feed-forward and the two RMS norm submodules.
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        # [FUNCTION DOC] Residual attention step on the normalized input followed by a residual
        # feed-forward step on its normalized result. Returns the layer output.
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    # [CLASS DOC] Multi-head attention with grouped key/value heads, rotary position embeddings and
    # an optional static KV cache (the kv_cache attribute, None until setup_caches). Query, key and
    # value use one fused projection. Registers a load hook to fuse legacy separate weights.
    def __init__(self, config: ModelArgs):
        # [FUNCTION DOC] Creates the fused qkv projection (sized for all query heads plus two sets
        # of key/value heads) and the output projection, both without bias, and copies head sizes
        # from the config. Raises AssertionError if the width is not divisible by the head count.
        # Registers the load_hook as a pre-hook for loading state dicts.
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
        # [FUNCTION DOC] State-dict pre-load hook. When the checkpoint holds separate query, key
        # and value weights under this module's prefix, removes them and stores their concatenation
        # (along the first dimension) as the fused qkv weight. Mutates the passed state dict;
        # otherwise a no-op.
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
        # [FUNCTION DOC] Computes attention over x of shape batch, sequence, width. Splits the
        # fused projection into query, key and value, reshapes to heads, applies rotary embeddings
        # to query and key, moves heads before sequence, updates the KV cache at input_pos if one
        # exists (using its full contents afterward), repeats key/value heads to match the query
        # heads, runs scaled dot-product attention with the given mask and no dropout, and applies
        # the output projection. Returns a tensor shaped like x.
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


class FeedForward(nn.Module):
    # [CLASS DOC] Gated (SwiGLU style) feed-forward: silu of one projection multiplied by a second
    # projection, then projected back to the model width. All three linears have no bias.
    def __init__(self, config: ModelArgs) -> None:
        # [FUNCTION DOC] Creates the three bias-free linear layers sized from the config.
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # [FUNCTION DOC] Returns the down projection of the silu-activated first projection times
        # the third projection of x.
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    # [CLASS DOC] Root-mean-square layer norm with a learned per-feature scale initialized to ones
    # and a small epsilon for numerical stability. Normalization is done in float32.
    def __init__(self, dim: int, eps: float = 1e-5):
        # [FUNCTION DOC] Stores eps and creates the learned scale parameter of the given size.
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # [FUNCTION DOC] HELPER: Divides x by the root mean square over the last dimension plus
        # eps.
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        # [FUNCTION DOC] Normalizes x in float32, casts back to the input dtype and multiplies by
        # the learned scale. Returns a tensor shaped like x.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    # [FUNCTION DOC] Builds the rotary position table for positions up to seq_len. Frequencies are
    # the inverse of base raised to even-index fractions of n_elem; they are multiplied by position
    # and turned into unit complex numbers, then the real and imaginary parts are stacked on a new
    # last axis. Returns a bfloat16 tensor shaped seq_len, n_elem / 2, 2.
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    # [FUNCTION DOC] Applies rotary embeddings to x shaped batch, sequence, heads, head dim. Treats
    # adjacent feature pairs as complex numbers, rotates them by the matching table entries (cos
    # and sin from freqs_cis, viewed to broadcast over batch and heads) in float32, flattens back,
    # and returns a tensor with the shape and dtype of x.
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
