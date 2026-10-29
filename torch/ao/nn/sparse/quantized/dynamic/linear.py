# mypy: allow-untyped-defs
# [MODULE DOC] Dynamically quantized sparse Linear module: Linear takes and
# returns float tensors, stores a block-sparse qint8 packed weight (via the
# sparse quantized LinearPackedParams) and runs the sparse dynamic quantized
# linear op. Includes state dict save/load, and from_float conversion that
# observes the (optionally masked) weight. No import-time side effects.

import torch
import torch.ao.nn.intrinsic as nni
from torch.ao.nn.quantized.modules.utils import (
    _hide_packed_params_repr,
    _quantize_weight,
)
from torch.ao.nn.sparse.quantized import linear
from torch.ao.nn.sparse.quantized.utils import LinearBlockSparsePattern


__all__ = ["Linear"]


class Linear(torch.nn.Module):
    # [CLASS DOC] Dynamically quantized block-sparse linear layer (float inputs
    # and outputs, qint8 weights). State: in_features, out_features and a
    # packed-params submodule holding the sparse packed weight and bias; class
    # attributes _version 1, _op_type sparse_dynamic and _FLOAT_MODULE (the
    # float Linear it converts from). Built via the constructor or from_float.
    r"""
    A dynamically quantized sparse linear module with float tensor as inputs and outputs.
    """

    _version = 1
    _op_type = "sparse_dynamic"
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
        # [FUNCTION DOC] Creates the layer with zero bias (when bias is true,
        # else none) and an empty qint8 weight, packs them with the given row
        # and column block sizes. Raises NotImplementedError if dtype is not
        # qint8.
        super().__init__()

        if dtype != torch.qint8:
            raise NotImplementedError(
                "Only QINT8 is supported for Sparse Quantized Linear Dynamic"
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
        self._packed_params = linear.LinearPackedParams(
            row_block_size=row_block_size, col_block_size=col_block_size, dtype=dtype
        )
        self._packed_params.set_weight_bias(
            qweight, bias, row_block_size, col_block_size
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # SparseQuantizedDynamicLinear.
        return "SparseQuantizedDynamicLinear"

    def extra_repr(self):
        # [FUNCTION DOC] HELPER: returns a string with in_features, out_features
        # and the weight's quantization scheme.
        return f"in_features={self.in_features}, out_features={self.out_features}, qscheme={self.weight().qscheme()}"

    def __repr__(self):
        # [FUNCTION DOC] HELPER: repr that hides the packed params submodule.
        return _hide_packed_params_repr(self, linear.LinearPackedParams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Applies the sparse dynamic quantized linear op to float
        # input x with the packed params; returns a float tensor.
        return torch.ops.sparse.qlinear_dynamic(x, self._packed_params._packed_params)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # [FUNCTION DOC] HELPER: Saves base state and adds the op type string
        # under the prefixed key op_type in the destination mapping.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "op_type"] = self._op_type

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
        # [FUNCTION DOC] HELPER: Loads state: reads and removes the op_type
        # entry, checks the version is not newer than supported
        # (AssertionError), moves the weight and bias entries into the packed
        # params submodule keys, then delegates to the base loader with strict
        # off. Mutates the state dict.
        # KNOWN ISSUE: the op type entry is converted to an integer and compared
        # to the string sparse, which can never match the saved op type
        # (sparse_dynamic), so loading always raises AssertionError (or fails on
        # conversion).
        op_type = int(state_dict[prefix + "op_type"])
        if op_type != "sparse":
            raise AssertionError(
                f"Cannot load from op_type [{op_type}], expecting [{self._op_type}]"
            )
        state_dict.pop(prefix + "op_type")

        version = local_metadata.get("version", None)
        if version is not None and version > self._version:
            raise AssertionError(f"version {version} > self._version {self._version}")

        # Is this code valid? In old quantization it seemed to be used to load
        # older model
        weight = state_dict.pop(prefix + "weight")
        bias = state_dict.pop(prefix + "bias")
        state_dict.update(
            {
                prefix + "_packed_params.weight": weight,
                prefix + "_packed_params.bias": bias,
            }
        )

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
        # [FUNCTION DOC] HELPER: Returns the unpacked (weight, bias) pair from
        # the packed params.
        return self._packed_params._weight_bias()

    def weight(self):
        # [FUNCTION DOC] Returns the unpacked quantized weight.
        return self._weight_bias()[0]

    def bias(self):
        # [FUNCTION DOC] Returns the unpacked bias (or None).
        return self._weight_bias()[1]

    def set_weight_bias(
        self,
        w: torch.Tensor,
        b: torch.Tensor | None,
        row_block_size: int | None,
        col_block_size: int | None,
    ) -> None:
        # [FUNCTION DOC] Updates out_features and in_features from the weight
        # shape and repacks weight w and bias b with the given block sizes.
        # Raises AssertionError if either block size is None.
        if row_block_size is None or col_block_size is None:
            raise AssertionError(
                f"row_block_size and col_block_size must not be None, got {row_block_size=}, {col_block_size=}"
            )
        self.out_features = w.shape[0]
        self.in_features = w.shape[1]
        self._packed_params.set_weight_bias(w, b, row_block_size, col_block_size)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Converts a float Linear (exact type required,
        # AssertionError otherwise; must have qconfig) into this module. Unwraps
        # a fused LinearReLU, gets a weight observer from the qconfig (or the
        # default dynamic qconfig, imported lazily to avoid a circular import),
        # multiplies the weight by the qconfig mask if present before observing,
        # requires qint8 and all-zero weight zero points (AssertionError
        # otherwise), quantizes the weight, reads the block size from the global
        # sparse pattern setting and returns a new module with packed weight and
        # bias.
        r"""Create a quantized sparse dynamic module from a float module.

        We only care about the convert at this stage, no need for observers just yet.
        """
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(
                " nnq."
                + cls.__name__
                + ".from_float only works for "
                + cls._FLOAT_MODULE.__name__
            )
        # TODO: Need to add options to qconfig to avoid the calibration.
        # TODO: Add calibration for the sparsity
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if type(mod) is nni.LinearReLU:
            mod = mod[0]
        # pyrefly: ignore [missing-attribute]
        if mod.qconfig is not None and mod.qconfig.weight is not None:
            # pyrefly: ignore [not-callable]
            weight_observer = mod.qconfig.weight()
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.ao.quantization.qconfig import default_dynamic_qconfig

            weight_observer = default_dynamic_qconfig.weight()

        # It is important to multiply by the mask BEFORE calling the `weight_observer`
        # TODO (zaf): Mask might not be part of the qconfig (T83295194)
        weight = mod.weight
        if getattr(mod.qconfig, "mask", False):
            weight = mod.qconfig.mask * mod.weight

        weight_observer(weight)
        dtype = weight_observer.dtype
        if dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have dtype torch.qint8, got {dtype}"
            )
        _w_sc, w_zp = weight_observer.calculate_qparams()
        if isinstance(w_zp, torch.Tensor):
            if torch.any(w_zp.bool()):
                raise AssertionError("All weight zero points must map to 0")
        else:
            if w_zp != 0:
                raise AssertionError(f"Weight zero point must map to 0, got {w_zp}")
        qweight = _quantize_weight(weight.float(), weight_observer)

        row_block_size, col_block_size = LinearBlockSparsePattern.block_size()
        qlinear = cls(
            mod.in_features,
            mod.out_features,
            row_block_size,
            col_block_size,
            dtype=dtype,
        )
        # pyrefly: ignore [bad-argument-type]
        qlinear.set_weight_bias(qweight, mod.bias, row_block_size, col_block_size)
        return qlinear
