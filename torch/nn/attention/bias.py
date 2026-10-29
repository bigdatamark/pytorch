# mypy: allow-untyped-defs
# [MODULE DOC]
# Causal attention bias types usable as the attention mask of scaled dot product attention:
# the CausalVariant enum, the CausalBias tensor subclass, and the factory functions
# causal_upper_left and causal_lower_right. Import-time side effect: registers several
# SDPA-related callables with dynamo as allowed in graph.
"""Defines bias subclasses that work with scaled_dot_product_attention"""

from enum import auto, IntEnum
from warnings import warn

import torch
import torch.nn.functional as F
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    is_flash_attention_available,
    SDPAParams,
)
from torch.nn.attention import _raise_kernel_warnings
from torch.nn.attention._utils import (
    _calculate_scale,
    _input_requires_grad,
    _postprocess_flash_output,
    _validate_sdpa_input,
)


__all__ = ["causal_upper_left", "causal_lower_right", "CausalVariant", "CausalBias"]


torch._dynamo.allow_in_graph(is_flash_attention_available)
torch._dynamo.allow_in_graph(can_use_flash_attention)
torch._dynamo.allow_in_graph(can_use_efficient_attention)
torch._dynamo.allow_in_graph(SDPAParams)


class CausalVariant(IntEnum):
    # [CLASS DOC]
    # Integer enum naming the two causal mask alignments: UPPER_LEFT (standard lower
    # triangular mask anchored at the top-left) and LOWER_RIGHT (anchored at the bottom-right
    # corner). Values are auto-numbered from 1; the integer value is passed to the
    # efficient attention kernel as its custom mask type.
    r"""
    Enum for causal variants used in attention mechanisms.

    Defines two types of causal biases:

    ``UPPER_LEFT``: Represents upper-left triangular bias for standard causal attention.
    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        torch.tril(torch.ones(size, dtype=torch.bool))

    For instance, with ``shape=(3,4)``, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0]]


    ``LOWER_RIGHT``: Represents lower-right triangular bias, the include values are aligned to the lower
    right corner of the matrix.

    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        diagonal_offset = size[1] - size[0]
        torch.tril(
            torch.ones(size, dtype=torch.bool),
            diagonal=diagonal_offset,
        )

    For instance, with ``shape=(3,4)``, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Note that these variants are equivalent to each other when the sequence lengths of the query and key/value
    tensors are equal since the triangular matrix is square.

    .. warning:: This enum is a prototype and subject to change.
    """

    UPPER_LEFT = auto()
    LOWER_RIGHT = auto()


class CausalBias(torch.Tensor):
    # [CLASS DOC]
    # Lightweight tensor subclass describing a causal mask lazily (no storage is allocated
    # for the mask). Attributes: variant (CausalVariant), seq_len_q, seq_len_kv. Its
    # __torch_function__ intercepts scaled_dot_product_attention calls and routes them to
    # _dispatch, which picks a fused kernel or falls back to a materialized boolean mask.
    # Other torch functions fall through to the default tensor behavior.
    """
    A bias representing causal attention patterns. For an overview of the bias structure, see the :class:`CausalVariant` enum.

    This class is used for defining causal (triangular) attention biases. For constructing the bias, there exist
    two factory functions: :func:`causal_upper_left` and :func:`causal_lower_right`.

    Example:

    .. code-block:: python

        from torch.nn.attention.bias import causal_lower_right

        bsz, num_heads, seqlen_q, seqlen_kv, head_dim = 32, 8, 4, 12, 8

        # Create a lower-right causal bias
        attn_bias = causal_lower_right(seqlen_q, seqlen_kv)

        q = torch.randn(
            bsz, num_heads, seqlen_q, head_dim, device="cuda", dtype=torch.float16
        )
        k = torch.randn(
            bsz, num_heads, seqlen_kv, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            bsz, num_heads, seqlen_kv, head_dim, device="cuda", dtype=torch.float16
        )

        out = F.scaled_dot_product_attention(q, k, v, attn_bias)

    .. warning:: This class is a prototype and subject to change.
    """

    def __init__(self, variant: CausalVariant, seq_len_q: int, seq_len_kv: int) -> None:
        # [FUNCTION DOC]
        # Store the variant and the query/key-value sequence lengths on the instance. Raises
        # AssertionError if variant is not a CausalVariant. Emits a warning when variant is
        # LOWER_RIGHT and seq_len_q exceeds seq_len_kv (outputs may contain NaNs).
        """
        Initializes the CausalBias instance with a specified variant and sequence lengths.

        Args:
            variant (CausalVariant): The type of causal bias to use (either UPPER_LEFT or LOWER_RIGHT).
            seq_len_q (int): The sequence length of the query tensor.
            seq_len_kv (int): The sequence length of the key/value tensor.

        Raises a warning if the LOWER_RIGHT variant is used with seq_len_q > seq_len_kv, as it may produce NaNs.
        """
        if not isinstance(variant, CausalVariant):
            raise AssertionError(
                f"variant must be a CausalVariant, got {type(variant).__name__}"
            )
        super().__init__()
        self.variant = variant
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        if seq_len_q > seq_len_kv and variant == CausalVariant.LOWER_RIGHT:
            warn(
                "Lower right causal bias will produce NaNs in the output when seq_len_q > seq_len_kv!",
                stacklevel=2,
            )

    def _upper_left(self, device: torch.device) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: build a boolean lower-triangular matrix of shape
        # (seq_len_q, seq_len_kv) on the given device (diagonal offset zero).
        """Upper left causal bias"""
        return torch.tril(
            torch.ones(self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool)
        )

    def _lower_right(self, device: torch.device) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: build a boolean lower-triangular matrix of shape
        # (seq_len_q, seq_len_kv) on the given device, with diagonal offset seq_len_kv minus
        # seq_len_q so the true region is aligned to the bottom-right corner.
        """Lower right causal bias"""
        diagonal_offset = self.seq_len_kv - self.seq_len_q
        return torch.tril(
            torch.ones(
                self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool
            ),
            diagonal=diagonal_offset,
        )

    # pyrefly: ignore [bad-return]
    def _materialize(self, device: torch.device | None = None) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: produce the dense boolean mask for this bias on device
        # (default CPU), choosing the upper-left or lower-right construction by variant.
        # Implicitly returns None for an unknown variant (hence the type-ignore above).
        """
        Materializes the causal bias into a tensor form.

        Depending on the variant, this method generates either an upper-left or lower-right
        triangular matrix to represent the causal bias.

        Args:
            device (Optional[torch.device]): The device on which to create the tensor. Defaults to CPU.

        Returns:
            torch.Tensor: The materialized bias tensor.
        """
        if device is None:
            device = torch.device("cpu")
        if self.variant == CausalVariant.UPPER_LEFT:
            return self._upper_left(device)
        elif self.variant == CausalVariant.LOWER_RIGHT:
            return self._lower_right(device)

    @staticmethod
    def _dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: "CausalBias",
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Static implementation of scaled dot product attention for a CausalBias mask.
        # Raises ValueError if is_causal is true, or if the variant is unknown. If the
        # sequence lengths are equal or the variant is upper-left, calls standard SDPA with
        # the causal flag and no mask. For lower-right: validates inputs, then uses the flash
        # kernel if allowed (padding the head dimension up to a multiple of 8, or 1 on xpu,
        # and unpadding the output), else the efficient attention kernel with the variant as
        # mask type, else standard SDPA with the materialized mask (after optional warnings).
        # Returns the attention output tensor.
        # KNOWN ISSUE: enable_gqa is ignored by the flash and efficient kernel paths in the
        # lower-right case (only the materialized-mask fallback honors it).
        r"""
        Handles the logic for computing attention with the specified causal bias.

        Args:
            query (Tensor): Query tensor; shape :math:`(N, ..., L, E)`.
            key (Tensor): Key tensor; shape :math:`(N, ..., S, E)`.
            value (Tensor): Value tensor; shape :math:`(N, ..., S, Ev)`.
            attn_mask (CausalBias): The type of causal attention to apply.
                A boolean mask where a value of True indicates that the element *should* take part in attention.
                A float mask of the same type as query, key, value that is added to the attention score.
            dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
            is_causal (bool): If true, assumes upper left causal attention masking and errors if both attn_mask and is_causal
                are set.
            scale (optional float): Scaling factor applied prior to softmax. If None, the default value is set
                to :math:`\frac{1}{\sqrt{E}}`.
            enable_gqa (optional bool): If set to True, Grouped Query Attention (GQA) is enabled, by default it is set to False.

        Returns:
            output (Tensor): Attention output; shape :math:`(N, ..., L, Ev)`.

        Raises:
            ValueError: If the causal bias variant is not a CausalVariant type.

        """
        if is_causal:
            raise ValueError("CausalBias should not be used with causal=True")

        if (
            attn_mask.seq_len_q == attn_mask.seq_len_kv
            or attn_mask.variant == CausalVariant.UPPER_LEFT
        ):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        elif attn_mask.variant == CausalVariant.LOWER_RIGHT:
            _validate_sdpa_input(query, key, value, None, dropout_p, is_causal, scale)
            sdpa_params = SDPAParams(
                query, key, value, None, dropout_p, is_causal, enable_gqa
            )
            if can_use_flash_attention(sdpa_params):
                alignment = 1 if query.device.type == "xpu" else 8
                og_head_size = query.size(-1)
                og_scale = _calculate_scale(og_head_size, scale)
                needs_padding = og_head_size % alignment != 0
                if needs_padding:
                    pad_len = alignment - (og_head_size % alignment)
                    query = torch.nn.functional.pad(query, (0, pad_len))
                    key = torch.nn.functional.pad(key, (0, pad_len))
                    value = torch.nn.functional.pad(value, (0, pad_len))
                out = torch.ops.aten._scaled_dot_product_flash_attention(
                    query,
                    key,
                    value,
                    dropout_p,
                    is_causal=True,  # TODO: Flash accepts causal = True and for this particular op it means lower right
                    return_debug_mask=False,
                    scale=og_scale,
                )[0]
                return _postprocess_flash_output(out, og_head_size)
            if can_use_efficient_attention(sdpa_params):
                compute_log_sumexp = False
                if _input_requires_grad(query, key, value):
                    compute_log_sumexp = True
                return torch.ops.aten._efficient_attention_forward(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    bias=None,
                    cu_seqlens_q=None,
                    cu_seqlens_k=None,
                    max_seqlen_q=None,
                    max_seqlen_k=None,
                    dropout_p=dropout_p,
                    custom_mask_type=int(attn_mask.variant),
                    compute_log_sumexp=compute_log_sumexp,
                    scale=scale,
                    seqlen_k=None,
                )[0].transpose(1, 2)
            else:
                _raise_kernel_warnings(sdpa_params)
                # We can't use efficient attention the only support for lower right is via materialization
                return F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask._materialize(query.device),
                    dropout_p=dropout_p,
                    is_causal=False,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
        else:
            raise ValueError(
                f"CausalBias.variant must be a CausalVariant type, but found: {attn_mask.variant}"
            )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Intercept torch function calls: when func is scaled dot product
        # attention, forward all arguments to _dispatch and return its result; otherwise
        # defer to the base tensor implementation. kwargs defaults to an empty mapping.
        """Defines the behavior of torch.nn.functional.scaled_dot_product_attention when the attn_bias is an AttnBias"""
        if kwargs is None:
            kwargs = {}
        if func is torch.nn.functional.scaled_dot_product_attention:
            return cls._dispatch(*args, **kwargs)
        return super().__torch_function__(func, types, args, kwargs)

    def __repr__(self) -> str:  # type:ignore[override]
        # [FUNCTION DOC] Return the repr of the materialized (CPU) boolean mask tensor.
        return self._materialize().__repr__()


def causal_upper_left(*size) -> CausalBias:
    # [FUNCTION DOC] Factory taking two sizes (query length, key/value length); returns a
    # CausalBias with the UPPER_LEFT variant. Raises AssertionError unless exactly two sizes are given.
    """
    Creates an upper-left triangular causal bias.

    This function generates an upper-left triangular matrix to represent causal attention bias with a
    diagonal offset set so that the inclusive values are aligned to the upper left corner of the matrix.
    This equivalent to the `is_causal=True` argument in `scaled_dot_product_attention`.

    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        torch.tril(torch.ones(size, dtype=torch.bool))

    For instance, with `shape=(3,4)`, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0]]

    Args:
        size: The size of the bias matrix.

    Returns:
        CausalBias: The UPPER_LEFT triangular causal bias variant.
    """
    if len(size) != 2:
        raise AssertionError("causal_upper_left only supports 2D tensors")
    seq_len_q, seq_len_kv = size
    return CausalBias(CausalVariant.UPPER_LEFT, seq_len_q, seq_len_kv)


def causal_lower_right(*size) -> CausalBias:
    # [FUNCTION DOC] Factory taking two sizes (query length, key/value length); returns a
    # CausalBias with the LOWER_RIGHT variant. Raises AssertionError unless exactly two sizes
    # are given. May warn (from the constructor) when query length exceeds key/value length.
    """
    Creates a lower-right triangular causal bias.

    This function generates a lower-right triangular matrix to represent causal attention bias with a
    diagonal offset set so that the inclusive values are aligned to the lower right corner of the matrix.

    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        diagonal_offset = size[1] - size[0]
        torch.tril(
            torch.ones(size, dtype=torch.bool),
            diagonal=diagonal_offset,
        )

    For instance, with `shape=(3,4)`, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Args:
        size: The size of the bias matrix.

    Returns:
        CausalBias: The LOWER_RIGHT triangular causal bias variant.
    """
    if len(size) != 2:
        raise AssertionError("causal_lower_right only supports 2D tensors")
    seq_len_q, seq_len_kv = size
    return CausalBias(CausalVariant.LOWER_RIGHT, seq_len_q, seq_len_kv)
