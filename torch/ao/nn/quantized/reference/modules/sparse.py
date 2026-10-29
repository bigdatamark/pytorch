# mypy: allow-untyped-defs
# [MODULE DOC] Reference quantized Embedding and EmbeddingBag modules for FX
# graph mode quantization. Each keeps a float weight and runs the float
# functional embedding op on a quantize-dequantize round trip of the weight. No
# import-time side effects.
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import ReferenceQuantizedModule


__all__ = ["Embedding", "EmbeddingBag"]


class Embedding(nn.Embedding, ReferenceQuantizedModule):
    # [CLASS DOC] Reference quantized Embedding: subclass of the float Embedding
    # and ReferenceQuantizedModule. Holds the float weight and weight
    # quantization parameters; the output is float.
    """A reference quantized Embedding module that fits into the
    FX Graph Mode Quantization workflow, activation will be floating point Tensor,
    we will store floating point weight as well in the module, but in forward we'll
    quantize and dequantize the weight before running the floating point functional
    embedding operator.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Tensor | None = None,
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the float Embedding base with the given
        # arguments (including optional pre-supplied weight, device, dtype),
        # then sets up weight quantization parameters from weight_qparams
        # (mapping, may be None).
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            # pyrefly: ignore [bad-argument-type]
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedEmbedding(Reference).
        return "QuantizedEmbedding(Reference)"

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Looks up embeddings for the index tensor using the
        # quantize-dequantized weight and the module's padding, max norm, norm
        # type, gradient scaling and sparse settings; returns a float tensor.
        weight_quant_dequant = self.get_weight()
        return F.embedding(
            input,
            weight_quant_dequant,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    @classmethod
    def from_float(cls, mod, weight_qparams):
        # [FUNCTION DOC] Builds a reference Embedding from a float Embedding,
        # reusing its configuration, weight (not copied), device and dtype with
        # the given weight_qparams. Returns the new module.
        return cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.padding_idx,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.sparse,
            mod.weight,
            mod.weight.device,
            mod.weight.dtype,
            weight_qparams,
        )


class EmbeddingBag(nn.EmbeddingBag, ReferenceQuantizedModule):
    # [CLASS DOC] Reference quantized EmbeddingBag: subclass of the float
    # EmbeddingBag and ReferenceQuantizedModule. Holds the float weight and
    # weight quantization parameters; the output is float.
    """A reference quantized EmbeddingBag module that fits into the
    FX Graph Mode Quantization workflow, activation will be floating point Tensor,
    we will store floating point weight as well in the module, but in forward we'll
    quantize and dequantize the weight before running the floating point functional
    embedding operator.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: Tensor | None = None,
        include_last_offset: bool = False,
        padding_idx: int | None = None,
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the float EmbeddingBag base with the given
        # arguments (including mode, last-offset flag, padding index, optional
        # weight, device, dtype), then sets up weight quantization parameters
        # from weight_qparams (mapping, may be None).
        super().__init__(
            num_embeddings,
            embedding_dim,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            mode,
            sparse,
            _weight,
            include_last_offset,
            padding_idx,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedEmbedding(Reference) (same name as the Embedding reference
        # module).
        return "QuantizedEmbedding(Reference)"

    def forward(
        self,
        input: Tensor,
        offsets: Tensor | None = None,
        per_sample_weights: Tensor | None = None,
    ) -> Tensor:
        # [FUNCTION DOC] Computes bagged embeddings of input (with optional
        # offsets and per-sample weights) using the quantize-dequantized weight
        # and the module's mode, norm and padding settings; returns a float
        # tensor.
        weight_quant_dequant = self.get_weight()
        return F.embedding_bag(
            input,
            weight_quant_dequant,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )

    @classmethod
    def from_float(cls, mod, weight_qparams, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Builds a reference EmbeddingBag from a float
        # EmbeddingBag, reusing its configuration, weight (not copied), device
        # and dtype with the given weight_qparams. use_precomputed_fake_quant is
        # unused. Returns the new module.
        return cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.mode,
            mod.sparse,
            mod.weight,
            mod.include_last_offset,
            mod.padding_idx,
            mod.weight.device,
            mod.weight.dtype,
            weight_qparams,
        )
