# mypy: allow-untyped-defs
# [MODULE DOC] Block-sparse quantized (qint8) Linear modules. Defines LinearPackedParams
# (holds prepacked sparse weight/bias with state-dict and pickle support) and Linear
# (quantized sparse linear layer built from a float Linear via from_float).
# Relies on the torch.ops.sparse qlinear prepack/unpack/forward ops. No import-time side effects.

import torch
from torch.ao.nn.quantized.modules.utils import (
    _hide_packed_params_repr,
    _quantize_weight,
)


__all__ = ["LinearPackedParams", "Linear"]


# TODO (zaf): Inherit from `quantized.LinearPackedParams` (T83294430)
class LinearPackedParams(torch.nn.Module):
    # [CLASS DOC] Module that stores the prepacked block-sparse quantized weight and bias
    # of a sparse quantized Linear in an opaque packed object (_packed_params).
    # Only qint8 is supported. forward is an identity; the class exists to own the packed
    # state, serialize it to/from state dicts (unpacked form) and pickle it.
    _version = 1

    def __init__(self, row_block_size=1, col_block_size=4, dtype=torch.qint8):
        # [FUNCTION DOC] Creates the module with a placeholder 1x1 quantized weight and no
        # bias, packed with the given block sizes. Raises NotImplementedError if dtype is
        # not qint8. Sets self.dtype and self._packed_params.
        super().__init__()

        if dtype != torch.qint8:
            raise NotImplementedError("Linear prepacking only supports QINT8")
        self.dtype = dtype
        wq = torch._empty_affine_quantized(
            [1, 1], scale=1.0, zero_point=0, dtype=torch.qint8
        )
        self.set_weight_bias(wq, None, row_block_size, col_block_size)

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name used in the module repr.
        return "SparseQuantizedLinearPackedParams"

    @torch.jit.export
    def set_weight_bias(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        row_block_size: int | None,
        col_block_size: int | None,
    ) -> None:
        # [FUNCTION DOC] Packs the quantized weight and optional bias into the sparse
        # prepacked format and stores it in _packed_params. Both block sizes are
        # required; raises AssertionError if either is None.
        if row_block_size is None or col_block_size is None:
            raise AssertionError(
                "row_block_size and col_block_size must not be None, got "
                f"row_block_size={row_block_size}, col_block_size={col_block_size}"
            )
        self._packed_params = torch.ops.sparse.qlinear_prepack(
            weight, bias, row_block_size, col_block_size
        )

    @torch.jit.export
    def _weight_bias(self):
        # [FUNCTION DOC] HELPER: Unpacks the stored params; returns a tuple of weight,
        # bias, row block size and column block size.
        (weight, bias, block_sizes) = torch.ops.sparse.qlinear_unpack(
            self._packed_params
        )
        return (weight, bias, block_sizes[0], block_sizes[1])

    def forward(self, x):
        # [FUNCTION DOC] Identity; returns the input unchanged (not a real computation).
        return x

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # [FUNCTION DOC] HELPER: Writes the dtype and the unpacked (weight, bias, row block,
        # col block) tuple into the destination mapping under prefixed keys.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "dtype"] = self.dtype
        destination[prefix + "_packed_params"] = self._weight_bias()

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
        # [FUNCTION DOC] HELPER: Restores dtype and repacks the weight/bias/block sizes popped
        # from the state dict, then delegates remaining keys to the base class with strict
        # disabled. Raises AssertionError if the saved version is newer than supported;
        # a missing key raises KeyError.
        version = local_metadata.get("version", None)
        if version is not None and version > self._version:
            raise AssertionError(f"version {version} > self._version {self._version}")

        self.dtype = state_dict.pop(prefix + "dtype")
        weight, bias, row_block_size, col_block_size = state_dict.pop(
            prefix + "_packed_params"
        )
        self.set_weight_bias(weight, bias, row_block_size, col_block_size)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def __getstate__(self):
        # [FUNCTION DOC] Returns the pickle state: packed params, training flag, dtype.
        return self._packed_params, self.training, self.dtype

    @torch.jit.export
    def __setstate__(self, state):
        # [FUNCTION DOC] Restores packed params, training flag and dtype from the state tuple.
        (self._packed_params, self.training, self.dtype) = state

    def __repr__(self):
        # [FUNCTION DOC] Returns the repr of the unpacked weight/bias/block-size tuple.
        return self._weight_bias().__repr__()


# TODO (zaf): Inherit from `quantized.Linear` (T83294430)
class Linear(torch.nn.Module):
    # [CLASS DOC] Quantized block-sparse qint8 linear layer. Holds in_features/out_features,
    # a LinearPackedParams (_packed_params) with the packed weight/bias, and the output
    # activation scale and zero_point. Created from a float torch.nn.Linear (_FLOAT_MODULE)
    # through from_float; forward calls the sparse qlinear op.
    r"""
    A quantized sparse linear module with quantized tensor as inputs and outputs.
    """

    _version = 1
    _FLOAT_MODULE = torch.nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        row_block_size,
        col_block_size,
        bias=True,
        dtype=torch.qint8,
    ):
        # [FUNCTION DOC] Builds the layer with a placeholder weight (scale 1, zero point 0)
        # and zero bias (if bias is true), packed with the given block sizes; scale=1.0,
        # zero_point=0. Raises NotImplementedError if dtype is not qint8.
        super().__init__()

        if dtype != torch.qint8:
            raise NotImplementedError(
                "Only QINT8 is supported for Sparse Quantized Linear"
            )

        self.in_features = in_features
        self.out_features = out_features

        if bias:
            bias = torch.zeros(self.out_features, dtype=torch.float)
        else:
            bias = None

        qweight = torch._empty_affine_quantized(
            [out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8
        )
        self._packed_params = LinearPackedParams(
            row_block_size=row_block_size, col_block_size=col_block_size, dtype=dtype
        )
        self._packed_params.set_weight_bias(
            qweight, bias, row_block_size, col_block_size
        )
        self.scale = 1.0
        self.zero_point = 0

    @classmethod
    def _get_name(cls):
        # [FUNCTION DOC] HELPER: Returns the display name used in the module repr.
        return "SparseQuantizedLinear"

    def extra_repr(self):
        # [FUNCTION DOC] Returns a string with features, scale, zero point and weight qscheme.
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, scale={self.scale}, "
            f"zero_point={self.zero_point}, qscheme={self.weight().qscheme()}"
        )

    def __repr__(self):
        # [FUNCTION DOC] Returns the module repr with the packed-params submodule hidden.
        return _hide_packed_params_repr(self, LinearPackedParams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Applies the sparse quantized linear op to a quantized input using
        # the packed params, output scale and zero point; returns a quantized tensor.
        return torch.ops.sparse.qlinear(
            x, self._packed_params._packed_params, self.scale, self.zero_point
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # [FUNCTION DOC] HELPER: Adds scale and zero_point (as tensors) to the destination
        # mapping after the base class entries.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = torch.tensor(self.scale)
        destination[prefix + "zero_point"] = torch.tensor(self.zero_point)

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
        # [FUNCTION DOC] HELPER: Restores scale and zero_point from the state dict, discards
        # the op_type entry (KeyError if missing), checks the saved version (AssertionError
        # if newer than supported), then loads remaining keys with strict disabled.
        self.scale = float(state_dict[prefix + "scale"])
        state_dict.pop(prefix + "scale")

        self.zero_point = int(state_dict[prefix + "zero_point"])
        state_dict.pop(prefix + "zero_point")

        state_dict.pop(prefix + "op_type")

        version = local_metadata.get("version", None)
        if version is not None and version > self._version:
            raise AssertionError(f"version {version} > self._version {self._version}")

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _weight_bias(self):
        # [FUNCTION DOC] HELPER: Returns the unpacked (weight, bias, row block, col block).
        return self._packed_params._weight_bias()

    def weight(self):
        # [FUNCTION DOC] Returns the unpacked quantized weight tensor.
        return self._weight_bias()[0]

    def bias(self):
        # [FUNCTION DOC] Returns the unpacked bias tensor (may be None).
        return self._weight_bias()[1]

    def set_weight_bias(
        self,
        w: torch.Tensor,
        b: torch.Tensor | None,
        row_block_size: int | None,
        col_block_size: int | None,
    ) -> None:
        # [FUNCTION DOC] Repacks the given weight and bias into the packed params with the
        # given block sizes. Raises AssertionError if either block size is None.
        if row_block_size is None or col_block_size is None:
            raise AssertionError(
                "row_block_size and col_block_size must not be None, "
                f"got row_block_size={row_block_size}, col_block_size={col_block_size}"
            )
        self._packed_params.set_weight_bias(w, b, row_block_size, col_block_size)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Converts a float Linear (exact type required) that has sparse_params
        # with a two-element sparse_block_shape, a qconfig and an activation_post_process
        # into a quantized sparse Linear. Runs the weight observer, requires qint8 and all
        # weight zero points equal to 0, quantizes the weight, builds the layer using the
        # block shape, and sets scale/zero_point from the activation observer. Raises
        # AssertionError on any violated precondition. use_precomputed_fake_quant is unused.
        r"""Create a quantized sparse module from a float module.

        We only care about the convert at this stage, no need for observers just yet.

        TODO(zaf): Need to add the sparse params to the qconfig
        """
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(
                cls._get_name()
                + ".from_float only works for "
                + cls._FLOAT_MODULE.__name__
            )
        if not hasattr(mod, "sparse_params"):
            raise AssertionError(
                "Expecting the Linear to have `sparse_params`. Make sure you have provided arguments "
                'in the `sparsifier.squash_mask(params_to_save=("sparse_block_shape",))` method.'
            )
        sparse_block_shape = mod.sparse_params.get("sparse_block_shape", None)  # type: ignore[operator, union-attr]
        if not isinstance(sparse_block_shape, (tuple, list)):
            raise AssertionError(
                f"sparse_block_shape must be tuple or list, got {type(sparse_block_shape)}"
            )
        if len(sparse_block_shape) != 2:
            raise AssertionError(
                f"sparse_block_shape must have length 2, got {len(sparse_block_shape)}"
            )
        # TODO: Need to add options to qconfig to avoid the calibration.
        # TODO: Add calibration for the sparsity
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        activation_post_process = mod.activation_post_process
        weight_post_process = mod.qconfig.weight()  # type: ignore[operator, union-attr]

        # Assumption is that the weight is already sparsified by the
        # `sparsifier.convert`
        weight = mod.weight

        weight_post_process(weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()  # type: ignore[operator, union-attr]
        if dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have dtype torch.qint8, got {dtype}"
            )
        w_sc, w_zp = weight_post_process.calculate_qparams()
        if isinstance(w_zp, torch.Tensor):
            if torch.any(w_zp.bool()):
                raise AssertionError("All weight zero points must map to 0")
        else:
            if w_zp != 0:
                raise AssertionError(f"Weight zero point must map to 0, got {w_zp}")
        qweight = _quantize_weight(weight.float(), weight_post_process)

        row_block_size = mod.sparse_params["sparse_block_shape"][0]  # type: ignore[index]
        col_block_size = mod.sparse_params["sparse_block_shape"][1]  # type: ignore[index]
        qlinear = cls(
            mod.in_features,
            mod.out_features,
            row_block_size,
            col_block_size,
            dtype=dtype,
        )
        qlinear.set_weight_bias(
            qweight,
            mod.bias,
            row_block_size,  # type: ignore[arg-type]
            col_block_size,  # type: ignore[arg-type]
        )
        qlinear.scale = float(act_scale)
        qlinear.zero_point = int(act_zp)
        return qlinear
