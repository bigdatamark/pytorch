# mypy: allow-untyped-defs
# [MODULE DOC] Wrapper modules that let functional ops (add, mul, cat, add_relu, matmul, scalar variants)
# be observed and quantized: FloatFunctional (float ops followed by an observer), FXFloatFunctional
# (plain float ops for FX mode, no observer), and QFunctional (quantized kernels with stored qparams).
# None of them supports forward; callers use the named op methods.

import torch
from torch import Tensor
from torch._ops import ops


__all__ = ["FloatFunctional", "FXFloatFunctional", "QFunctional"]


class FloatFunctional(torch.nn.Module):
    # [CLASS DOC] Collects activation statistics for float functional ops. Holds activation_post_process
    # (initially Identity, later replaced by an observer) that is applied to the result of most ops.
    # forward always raises; use add, mul, cat, add_relu, matmul, add_scalar, mul_scalar.
    r"""State collector class for float operations.

    The instance of this class can be used instead of the ``torch.`` prefix for
    some operations. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> f_add = FloatFunctional()
        >>> a = torch.tensor(3.0)
        >>> b = torch.tensor(4.0)
        >>> f_add.add(a, b)  # Equivalent to ``torch.add(a, b)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the module and sets activation_post_process to an Identity module.
        super().__init__()
        self.activation_post_process = torch.nn.Identity()

    def forward(self, x):
        # [FUNCTION DOC] Always raises RuntimeError: this class has no forward; use one of the named operations.
        raise RuntimeError(
            "FloatFunctional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

    r"""Operation equivalent to ``torch.add(Tensor, Tensor)``"""

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        # [FUNCTION DOC] Adds two tensors, passes the result through activation_post_process (observed) and returns it.
        r = torch.add(x, y)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.add(Tensor, float)``"""

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        # [FUNCTION DOC] Adds a scalar to a tensor and returns the result; deliberately not observed.
        r = torch.add(x, y)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, Tensor)``"""

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        # [FUNCTION DOC] Multiplies two tensors, passes the result through activation_post_process and returns it.
        r = torch.mul(x, y)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, float)``"""

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        # [FUNCTION DOC] Multiplies a tensor by a scalar and returns the result; deliberately not observed.
        r = torch.mul(x, y)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.cat``"""

    def cat(self, x: list[Tensor], dim: int = 0) -> Tensor:
        # [FUNCTION DOC] Concatenates a list of tensors along dim (default 0), observes the result and returns it.
        r = torch.cat(x, dim=dim)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``relu(torch.add(x,y))``"""

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        # [FUNCTION DOC] Adds two tensors, applies relu, observes the result and returns it.
        r = torch.add(x, y)
        r = torch.nn.functional.relu(r)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.matmul(Tensor, Tensor)``"""

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        # [FUNCTION DOC] Matrix-multiplies two tensors, observes the result and returns it.
        r = torch.matmul(x, y)
        r = self.activation_post_process(r)
        return r


class FXFloatFunctional(torch.nn.Module):
    # [CLASS DOC] Stateless replacement for FloatFunctional used before FX graph mode quantization, since
    # observers are inserted into the top-level module there. Ops are plain float ops with no observer.
    # forward always raises.
    r"""module to replace FloatFunctional module before FX graph mode quantization,
    since activation_post_process will be inserted in top level module directly

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def forward(self, x):
        # [FUNCTION DOC] Always raises RuntimeError: use one of the named operations instead.
        raise RuntimeError(
            "FloatFunctional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

    r"""Operation equivalent to ``torch.add(Tensor, Tensor)``"""

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        # [FUNCTION DOC] Returns the sum of two tensors.
        r = torch.add(x, y)
        return r

    r"""Operation equivalent to ``torch.add(Tensor, float)``"""

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        # [FUNCTION DOC] Returns the tensor plus a scalar.
        r = torch.add(x, y)
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, Tensor)``"""

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        # [FUNCTION DOC] Returns the product of two tensors.
        r = torch.mul(x, y)
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, float)``"""

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        # [FUNCTION DOC] Returns the tensor times a scalar.
        r = torch.mul(x, y)
        return r

    r"""Operation equivalent to ``torch.cat``"""

    def cat(self, x: list[Tensor], dim: int = 0) -> Tensor:
        # [FUNCTION DOC] Returns the concatenation of the tensor list along dim (default 0).
        r = torch.cat(x, dim=dim)
        return r

    r"""Operation equivalent to ``relu(torch.add(x,y))``"""

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        # [FUNCTION DOC] Returns relu applied to the sum of two tensors.
        r = torch.add(x, y)
        r = torch.nn.functional.relu(r)
        return r

    r"""Operation equivalent to ``torch.matmul(Tensor, Tensor)``"""

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        # [FUNCTION DOC] Returns the matrix product of two tensors.
        r = torch.matmul(x, y)
        return r


class QFunctional(torch.nn.Module):
    # [CLASS DOC] Wrapper that calls quantized kernels (torch.ops.quantized) for add, mul, cat, add_relu,
    # matmul and scalar variants. State: scale (float, default 1.0), zero_point (int, default 0) used as
    # output qparams, and activation_post_process (Identity by default). scale and zero_point are saved and
    # loaded through the state dict as tensors. forward always raises. Created from a FloatFunctional via from_float.
    r"""Wrapper class for quantized operations.

    The instance of this class can be used instead of the
    ``torch.ops.quantized`` prefix. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> q_add = QFunctional()
        >>> # xdoctest: +SKIP
        >>> a = torch.quantize_per_tensor(torch.tensor(3.0), 1.0, 0, torch.qint32)
        >>> b = torch.quantize_per_tensor(torch.tensor(4.0), 1.0, 0, torch.qint32)
        >>> q_add.add(a, b)  # Equivalent to ``torch.ops.quantized.add(a, b, 1.0, 0)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes output scale 1.0, zero point 0 and an Identity activation_post_process.
        super().__init__()
        self.scale = 1.0
        self.zero_point = 0
        self.activation_post_process = torch.nn.Identity()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # [FUNCTION DOC] HELPER: Extends the standard state dict saving by adding scale and zero_point as
        # tensors under the prefixed keys. Mutates destination.
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
        # [FUNCTION DOC] HELPER: Pops the prefixed scale and zero_point entries from state_dict into the
        # attributes (as float and int), then runs the standard loading with strict checking disabled.
        # Raises KeyError if the entries are absent.
        self.scale = float(state_dict.pop(prefix + "scale"))
        self.zero_point = int(state_dict.pop(prefix + "zero_point"))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QFunctional".
        return "QFunctional"

    def extra_repr(self):
        # [FUNCTION DOC] HELPER: Returns a string with scale and zero_point for the module repr.
        return f"scale={self.scale}, zero_point={self.zero_point}"

    def forward(self, x):
        # [FUNCTION DOC] Always raises RuntimeError: use one of the named operations instead.
        raise RuntimeError(
            "Functional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

    r"""Operation equivalent to ``torch.ops.quantized.add``"""

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        # [FUNCTION DOC] Quantized add of two tensors using the stored scale and zero_point; result is passed through activation_post_process.
        r = ops.quantized.add(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.ops.quantized.add(Tensor, float)``"""

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        # [FUNCTION DOC] Quantized add of a tensor and a scalar; returns the result without observation.
        r = ops.quantized.add_scalar(x, y)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.ops.quantized.mul(Tensor, Tensor)``"""

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        # [FUNCTION DOC] Quantized multiply of two tensors with the stored scale and zero_point; result passes through activation_post_process.
        r = ops.quantized.mul(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.ops.quantized.mul(Tensor, float)``"""

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        # [FUNCTION DOC] Quantized multiply by a scalar; returns the result without observation.
        r = ops.quantized.mul_scalar(x, y)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.ops.quantized.cat``"""

    def cat(self, x: list[Tensor], dim: int = 0) -> Tensor:
        # [FUNCTION DOC] Quantized concatenation of a tensor list along dim (default 0) using the stored qparams; result passes through activation_post_process.
        r = ops.quantized.cat(x, scale=self.scale, zero_point=self.zero_point, dim=dim)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.ops.quantized.add_relu``"""

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        # [FUNCTION DOC] Quantized fused add and relu with the stored qparams; result passes through activation_post_process.
        r = ops.quantized.add_relu(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.ops.quantized.matmul(Tensor, Tensor)``"""

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        # [FUNCTION DOC] Quantized matrix multiply with the stored qparams; returns the result without observation
        # (despite the placeholder observer being present).
        r = ops.quantized.matmul(x, y, scale=self.scale, zero_point=self.zero_point)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Class method: converts a FloatFunctional (exact type required, else AssertionError is
        # raised) into a QFunctional whose scale and zero_point come from the float module's observer.
        if type(mod) is not FloatFunctional:
            raise AssertionError(
                f"QFunctional.from_float expects an instance of FloatFunctional, "
                f"got {type(mod).__name__}"
            )
        scale, zero_point = mod.activation_post_process.calculate_qparams()  # type: ignore[operator]
        new_mod = QFunctional()
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod
