# mypy: allow-untyped-defs
# [MODULE DOC] Quantized embedding modules: EmbeddingPackedParams (holds the prepacked quantized weight
# and handles its serialization), Embedding and EmbeddingBag (lookup and bag-reduction using quantized
# kernels, for 8-bit or 4-bit packed weights). Provides conversion from float, QAT and reference modules.
import torch
import torch.nn as nn
from torch import Tensor
from torch._jit_internal import List, Optional  # noqa: F401

from .utils import _hide_packed_params_repr, _quantize_weight


__all__ = ["EmbeddingPackedParams", "Embedding", "EmbeddingBag"]


class EmbeddingPackedParams(torch.nn.Module):
    # [CLASS DOC] Container for the packed quantized embedding weight. State: dtype (quint8 or quint4x2) and
    # _packed_weight (opaque prepacked tensor). State dict stores dtype and the unpacked weight under
    # prefixed keys. forward is an identity placeholder. Version 1 of the serialization format.
    _version = 1

    def __init__(self, num_embeddings, embedding_dim, dtype=torch.quint8):
        # [FUNCTION DOC] Records dtype and creates a default weight of ones scale / zero zero point per-channel
        # (axis 0) quantized empty tensor of the given shape, which is packed through set_weight.
        # Raises NotImplementedError if dtype is not quint8 or quint4x2.
        super().__init__()
        self.dtype = dtype
        if self.dtype in [torch.quint8, torch.quint4x2]:
            scales = torch.ones(num_embeddings, dtype=torch.float)
            zero_points = torch.zeros(num_embeddings, dtype=torch.float)
            wq = torch._empty_per_channel_affine_quantized(
                [num_embeddings, embedding_dim],
                scales=scales,
                zero_points=zero_points,
                axis=0,
                dtype=self.dtype,
            )
            self.set_weight(wq)
        else:
            raise NotImplementedError(
                f"Unsupported dtype on quantized embedding! Supports quint8 and quint4x2. Got dtype: {dtype}"
            )

    @torch.jit.export
    def set_weight(self, weight: torch.Tensor) -> None:
        # [FUNCTION DOC] Packs a quantized weight tensor with the embedding_bag prepack kernel and stores it in
        # _packed_weight (exported for TorchScript). Raises NotImplementedError for unsupported dtypes.
        if self.dtype in [torch.quint8, torch.quint4x2]:
            self._packed_weight = torch.ops.quantized.embedding_bag_prepack(weight)
        else:
            raise NotImplementedError(
                "Unsupported dtype for quantized embedding prepack! Supports quint8 and quint4x2."
            )

    @torch.jit.export
    def _weight(self):
        # [FUNCTION DOC] HELPER: Unpacks and returns the quantized weight from _packed_weight (exported for
        # TorchScript). Raises NotImplementedError for unsupported dtypes.
        if self.dtype in [torch.quint8, torch.quint4x2]:
            return torch.ops.quantized.embedding_bag_unpack(self._packed_weight)
        else:
            raise NotImplementedError(
                "Unsupported dtype for quantized embedding unpack! Supports quint8 and quint4x2."
            )

    def forward(self, x):
        # [FUNCTION DOC] Identity placeholder; returns x unchanged.
        return x

    # Version 1
    #   self
    #   |--- _packed_weight : Tensor representing weight of EmbeddingPackedParamsBase
    #   |--- dtype : torch.dtype

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # [FUNCTION DOC] HELPER: Extends state dict saving with the dtype and the unpacked weight under prefixed keys.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "dtype"] = self.dtype
        destination[prefix + "_packed_weight"] = self._weight()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # [FUNCTION DOC] HELPER: Pops the prefixed dtype and weight entries from state_dict, sets dtype, repacks the
        # weight with set_weight, then runs the standard loading with strict checking disabled.
        # Raises KeyError if the entries are missing.
        self.dtype = state_dict[prefix + "dtype"]
        state_dict.pop(prefix + "dtype")

        weight = state_dict[prefix + "_packed_weight"]
        state_dict.pop(prefix + "_packed_weight")
        self.set_weight(weight)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns the repr of the unpacked weight tensor.
        return self._weight().__repr__()


class Embedding(torch.nn.Module):
    # [CLASS DOC] Quantized Embedding with a prepacked weight (stored in an EmbeddingPackedParams submodule
    # named _packed_params). Attributes: num_embeddings, embedding_dim, dtype. Same interface as the float
    # Embedding but forward only does lookups. Built from float/QAT modules (from_float) or reference modules (from_reference).
    r"""
    A quantized Embedding module with quantized packed weights as inputs.
    We adopt the same interface as `torch.nn.Embedding`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html for documentation.

    Similar to :class:`~torch.nn.Embedding`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{num\_embeddings}, \text{embedding\_dim})`.

    Examples::
        >>> warnings.filterwarnings("ignore", message=".*quantized tensor creation functions")  # docs: hide
        >>> m = nn.quantized.Embedding(num_embeddings=10, embedding_dim=12)
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8])
        >>> output = m(indices)
        >>> print(output.size())
        torch.Size([9, 12])

    """

    _version = 1

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        dtype=torch.quint8,
    ) -> None:
        # [FUNCTION DOC] Stores sizes and dtype, builds the weight (a default per-channel quantized quint8 tensor
        # when _weight is None, otherwise the supplied _weight whose shape must equal num_embeddings by
        # embedding_dim, else AssertionError), and stores it packed. padding_idx, max_norm, norm_type,
        # scale_grad_by_freq and sparse are accepted but unused.
        # KNOWN ISSUE: with dtype quint4x2 and no _weight, the default weight is still created as quint8.
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        if _weight is None:
            scales = torch.ones(num_embeddings, dtype=torch.float)
            zero_points = torch.zeros(num_embeddings, dtype=torch.float)
            qweight = torch._empty_per_channel_affine_quantized(
                [num_embeddings, embedding_dim],
                scales=scales,
                zero_points=zero_points,
                axis=0,
                dtype=torch.quint8,
            )
        else:
            expected_shape = [num_embeddings, embedding_dim]
            if list(_weight.shape) != expected_shape:
                raise AssertionError(
                    f"Shape of weight does not match num_embeddings and embedding_dim: "
                    f"expected {expected_shape}, got {list(_weight.shape)}"
                )
            qweight = _weight

        self._packed_params = EmbeddingPackedParams(
            num_embeddings, embedding_dim, dtype
        )
        self._packed_params.set_weight(qweight)

    def forward(self, indices: Tensor) -> Tensor:
        # [FUNCTION DOC] Looks up the rows of the packed weight for the integer indices tensor using the 4-bit
        # kernel if dtype is quint4x2, else the byte kernel; returns a float tensor.
        if self.dtype == torch.quint4x2:
            return torch.ops.quantized.embedding_4bit(
                self._packed_params._packed_weight, indices
            )
        else:
            return torch.ops.quantized.embedding_byte(
                self._packed_params._packed_weight, indices
            )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedEmbedding".
        return "QuantizedEmbedding"

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a repr that hides the packed params submodule's repr.
        return _hide_packed_params_repr(self, EmbeddingPackedParams)

    def extra_repr(self):
        # [FUNCTION DOC] HELPER: Returns a string with num_embeddings, embedding_dim, dtype and the weight's qscheme.
        extra_repr_str = (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"dtype={self._packed_params.dtype}, qscheme={self.weight().qscheme()}"
        )

        return extra_repr_str

    def set_weight(self, w: torch.Tensor) -> None:
        # [FUNCTION DOC] Packs and stores the quantized weight tensor w in the packed params submodule.
        self._packed_params.set_weight(w)

    def weight(self):
        # [FUNCTION DOC] Returns the unpacked quantized weight tensor.
        return self._packed_params._weight()

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Class method: converts a float nn.Embedding (or a QAT Embedding when it has a weight fake
        # quant) to a quantized Embedding. The weight observer comes from the fake quant, else from the module's
        # qconfig weight factory, else from the default float-qparams weight-only qconfig. Runs the observer on the
        # weight, quantizes it and stores it. Raises AssertionError if the module type is wrong, a qconfig is
        # missing, the observer scheme is not per-channel affine with float qparams, or dtype is not quint8/quint4x2.
        # KNOWN ISSUE: the new module is built without the observer dtype, so quint4x2 weights are packed under a quint8 default.
        r"""Create a quantized embedding module from a float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by user
        """
        if hasattr(mod, "weight_fake_quant"):
            if type(mod) is not torch.ao.nn.qat.Embedding:
                raise AssertionError(
                    "nnq."
                    + cls.__name__
                    + ".from_float "
                    + "with fake quant only works for "
                    + torch.ao.nn.qat.Embedding.__name__
                )
            weight_observer = mod.weight_fake_quant
        else:
            if type(mod) is not nn.Embedding:
                raise AssertionError(
                    "nnq."
                    + cls.__name__
                    + ".from_float only works for "
                    + nn.Embedding.__name__
                )
            if not hasattr(mod, "qconfig"):
                raise AssertionError(
                    "Embedding input float module must have qconfig defined"
                )
            from torch.ao.quantization import float_qparams_weight_only_qconfig

            if mod.qconfig is not None and mod.qconfig.weight is not None:  # type: ignore[union-attr]
                weight_observer = mod.qconfig.weight()  # type: ignore[union-attr, operator]
            else:
                weight_observer = float_qparams_weight_only_qconfig.weight()

        dtype = weight_observer.dtype
        is_float_qparams_qconfig = (
            weight_observer.qscheme == torch.per_channel_affine_float_qparams
        )
        if not is_float_qparams_qconfig:
            raise AssertionError(
                "Embedding quantization is only supported with float_qparams_weight_only_qconfig."
            )

        if dtype != torch.quint8 and dtype != torch.quint4x2:
            raise AssertionError(
                f"The only supported dtype for nnq.Embedding is torch.quint8 and torch.quint4x2, got {dtype}"
            )

        # Run the observer to calculate qparams.
        weight_observer(mod.weight)
        qweight = _quantize_weight(mod.weight.float(), weight_observer)

        # Create quantized Embedding module and pass in the quantized weight
        qembedding = Embedding(mod.num_embeddings, mod.embedding_dim)
        qembedding.set_weight(qweight)
        return qembedding

    @classmethod
    def from_reference(cls, ref_embedding):
        # [FUNCTION DOC] Class method: builds a quantized Embedding from a reference embedding, reusing its
        # sizes, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, quantized weight and weight_dtype.
        qembedding = cls(
            ref_embedding.num_embeddings,
            ref_embedding.embedding_dim,
            ref_embedding.padding_idx,
            ref_embedding.max_norm,
            ref_embedding.norm_type,
            ref_embedding.scale_grad_by_freq,
            ref_embedding.sparse,
            ref_embedding.get_quantized_weight(),
            ref_embedding.weight_dtype,
        )
        return qembedding


class EmbeddingBag(Embedding):
    # [CLASS DOC] Quantized EmbeddingBag derived from Embedding. Adds mode (reduction), pruned_weights (False),
    # include_last_offset and dtype. forward reduces bags of indices using offsets with the byte or 4-bit kernels.
    r"""
    A quantized EmbeddingBag module with quantized packed weights as inputs.
    We adopt the same interface as `torch.nn.EmbeddingBag`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html for documentation.

    Similar to :class:`~torch.nn.EmbeddingBag`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{num\_embeddings}, \text{embedding\_dim})`.

    Examples::
        >>> m = nn.quantized.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True, mode='sum')
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        >>> offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        >>> output = m(indices, offsets)
        >>> print(output.size())
        torch.Size([5, 12])

    """

    _version = 1

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "sum",
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        include_last_offset: bool = False,
        dtype=torch.quint8,
    ) -> None:
        # [FUNCTION DOC] Initializes the Embedding base with sizes, optional weight and dtype, then stores mode,
        # include_last_offset and dtype, and sets pruned_weights to False. max_norm, norm_type,
        # scale_grad_by_freq and sparse are accepted but unused.
        super().__init__(num_embeddings, embedding_dim, _weight=_weight, dtype=dtype)

        self.mode = mode
        self.pruned_weights = False
        self.include_last_offset = include_last_offset
        self.dtype = dtype

    def forward(
        self,
        indices: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
        compressed_indices_mapping: Optional[Tensor] = None,
    ) -> Tensor:
        # [FUNCTION DOC] Computes the bag reductions of the packed weight for indices and offsets, with optional
        # per_sample_weights and compressed_indices_mapping, using the 4-bit kernel if dtype is quint4x2, else
        # the byte kernel. Returns a float tensor with one row per bag.
        if self.dtype == torch.quint4x2:
            return torch.ops.quantized.embedding_bag_4bit(
                self._packed_params._packed_weight,
                indices,
                offsets,
                False,
                0,
                self.pruned_weights,
                per_sample_weights,
                compressed_indices_mapping,
                self.include_last_offset,
            )
        else:
            return torch.ops.quantized.embedding_bag_byte(
                self._packed_params._packed_weight,
                indices,
                offsets,
                False,
                0,
                self.pruned_weights,
                per_sample_weights,
                compressed_indices_mapping,
                self.include_last_offset,
            )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedEmbeddingBag".
        return "QuantizedEmbeddingBag"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Class method: converts a float nn.EmbeddingBag (or a QAT module with a weight fake quant)
        # to a quantized EmbeddingBag, using the same observer selection and weight quantization as the Embedding
        # version, and passing max_norm, norm_type, scale_grad_by_freq, mode, sparse, include_last_offset and the
        # observer dtype. Raises AssertionError for wrong module type, missing qconfig, a non float-qparams
        # observer scheme or an unsupported dtype.
        r"""Create a quantized embedding_bag module from a float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by user
        """
        if hasattr(mod, "weight_fake_quant"):
            weight_observer = mod.weight_fake_quant
        else:
            if type(mod) is not nn.EmbeddingBag:
                raise AssertionError(
                    "nnq."
                    + cls.__name__
                    + ".from_float only works for "
                    + nn.EmbeddingBag.__name__
                )
            if not hasattr(mod, "qconfig"):
                raise AssertionError(
                    "EmbeddingBag input float module must have qconfig defined"
                )
            from torch.ao.quantization.qconfig import float_qparams_weight_only_qconfig

            if mod.qconfig is not None and mod.qconfig.weight is not None:  # type: ignore[union-attr]
                weight_observer = mod.qconfig.weight()  # type: ignore[union-attr, operator]
            else:
                weight_observer = float_qparams_weight_only_qconfig.weight()

        dtype = weight_observer.dtype
        is_float_qparams_qconfig = (
            weight_observer.qscheme == torch.per_channel_affine_float_qparams
        )
        if not is_float_qparams_qconfig:
            raise AssertionError(
                "EmbeddingBag quantization is only supported with float_qparams_weight_only_qconfig."
            )

        if dtype != torch.quint8 and dtype != torch.quint4x2:
            raise AssertionError(
                f"The only supported dtype for nnq.EmbeddingBag is torch.quint8 and torch.quint4x2, got {dtype}"
            )

        # Run the observer to calculate qparams.
        weight_observer(mod.weight)
        qweight = _quantize_weight(mod.weight.float(), weight_observer)

        # Create quantized EmbeddingBag module and pass in the quantized weight
        qembedding_bag = EmbeddingBag(
            mod.num_embeddings,
            mod.embedding_dim,
            max_norm=mod.max_norm,
            norm_type=mod.norm_type,
            scale_grad_by_freq=mod.scale_grad_by_freq,
            mode=mod.mode,
            sparse=mod.sparse,
            include_last_offset=mod.include_last_offset,
            dtype=dtype,
        )
        qembedding_bag.set_weight(qweight)
        return qembedding_bag

    @classmethod
    def from_reference(cls, ref_embedding_bag):
        # [FUNCTION DOC] Class method: builds a quantized EmbeddingBag from a reference module, reusing its sizes,
        # reduction settings, quantized weight, include_last_offset and weight_dtype.
        qembedding_bag = cls(
            ref_embedding_bag.num_embeddings,
            ref_embedding_bag.embedding_dim,
            ref_embedding_bag.max_norm,
            ref_embedding_bag.norm_type,
            ref_embedding_bag.scale_grad_by_freq,
            ref_embedding_bag.mode,
            ref_embedding_bag.sparse,
            ref_embedding_bag.get_quantized_weight(),
            ref_embedding_bag.include_last_offset,
            ref_embedding_bag.weight_dtype,
        )
        return qembedding_bag
