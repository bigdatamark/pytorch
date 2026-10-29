# mypy: allow-untyped-defs
# [MODULE DOC] Fused float modules used to mark operator sequences (conv/linear/batch norm/relu/add combinations) for quantization; each is a Sequential container that the quantization flow later replaces with a fused quantized or QAT module. Exports the public fused classes; _FusedModule is the private marker base.
import torch
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    Linear,
    ReLU,
)
from torch.nn.utils.parametrize import type_before_parametrizations


__all__ = [
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearReLU",
    "ConvBn1d",
    "ConvBn2d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBn3d",
    "ConvBnReLU3d",
    "BNReLU2d",
    "BNReLU3d",
    "LinearBn1d",
    "LinearLeakyReLU",
    "LinearTanh",
    "ConvAdd2d",
    "ConvAddReLU2d",
]


# Used for identifying intrinsic modules used in quantization
class _FusedModule(torch.nn.Sequential):
    # [CLASS DOC] HELPER: Marker base class (a torch.nn.Sequential) used to identify fused intrinsic modules during quantization; adds no behavior.
    pass


class ConvReLU1d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Conv1d, ReLU), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Conv1d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Conv1d, ReLU, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(conv) == Conv1d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, relu)


class ConvReLU2d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Conv2d, ReLU), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Conv2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Conv2d, ReLU, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, relu)


class ConvReLU3d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Conv3d, ReLU), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Conv3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Conv3d, ReLU, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(conv) == Conv3d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, relu)


class LinearReLU(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Linear, ReLU), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Linear and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, relu):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Linear, ReLU, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(linear) == Linear
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(linear).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(linear, relu)


class ConvBn1d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Conv1d, BatchNorm1d), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Conv 1d and Batch Norm 1d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Conv1d, BatchNorm1d, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(conv) == Conv1d
            and type_before_parametrizations(bn) == BatchNorm1d
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(bn).__name__}"
            )
        super().__init__(conv, bn)


class ConvBn2d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Conv2d, BatchNorm2d), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Conv2d, BatchNorm2d, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(bn) == BatchNorm2d
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(bn).__name__}"
            )
        super().__init__(conv, bn)


class ConvBnReLU1d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Conv1d, BatchNorm1d, ReLU), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Conv 1d, Batch Norm 1d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Conv1d, BatchNorm1d, ReLU, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(conv) == Conv1d
            and type_before_parametrizations(bn) == BatchNorm1d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__}, "
                f"{type_before_parametrizations(bn).__name__}, and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, bn, relu)


class ConvBnReLU2d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Conv2d, BatchNorm2d, ReLU), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Conv2d, BatchNorm2d, ReLU, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(bn) == BatchNorm2d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__}, "
                f"{type_before_parametrizations(bn).__name__}, and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, bn, relu)


class ConvBn3d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Conv3d, BatchNorm3d), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Conv 3d and Batch Norm 3d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Conv3d, BatchNorm3d, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(conv) == Conv3d
            and type_before_parametrizations(bn) == BatchNorm3d
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(bn).__name__}"
            )
        super().__init__(conv, bn)


class ConvBnReLU3d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Conv3d, BatchNorm3d, ReLU), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Conv3d, BatchNorm3d, ReLU, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(conv) == Conv3d
            and type_before_parametrizations(bn) == BatchNorm3d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__}, "
                f"{type_before_parametrizations(bn).__name__}, and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, bn, relu)


class BNReLU2d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (BatchNorm2d, ReLU), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the BatchNorm 2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, batch_norm, relu):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (BatchNorm2d, ReLU, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(batch_norm) == BatchNorm2d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(batch_norm).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(batch_norm, relu)


class BNReLU3d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (BatchNorm3d, ReLU), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the BatchNorm 3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, batch_norm, relu):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (BatchNorm3d, ReLU, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(batch_norm) == BatchNorm3d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(batch_norm).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(batch_norm, relu)


class LinearBn1d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Linear, BatchNorm1d), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Linear and BatchNorm1d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, bn):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Linear, BatchNorm1d, ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (
            type_before_parametrizations(linear) == Linear
            and type_before_parametrizations(bn) == BatchNorm1d
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(linear).__name__} and "
                f"{type_before_parametrizations(bn).__name__}"
            )
        super().__init__(linear, bn)


class LinearLeakyReLU(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Linear, LeakyReLU (exact types)), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Linear and LeakyReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, leaky_relu):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Linear, LeakyReLU (exact types), ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (type(linear) is Linear and type(leaky_relu) is torch.nn.LeakyReLU):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type(linear).__name__} and {type(leaky_relu).__name__}"
            )
        super().__init__(linear, leaky_relu)


class LinearTanh(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Linear, Tanh (exact types)), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Linear and Tanh modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, tanh):
        # [FUNCTION DOC] Validate that each argument has exactly the expected type (Linear, Tanh (exact types), ignoring parametrizations where checked) and build the Sequential from them in order. Raises AssertionError listing the actual type names otherwise.
        if not (type(linear) is Linear and type(tanh) is torch.nn.Tanh):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type(linear).__name__} and {type(tanh).__name__}"
            )
        super().__init__(linear, tanh)


class ConvAdd2d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Conv2d, add module), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Conv2d modules with extra Add.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, add):
        # [FUNCTION DOC] Build the container from the conv module (kept as the sequential child) and store the add module as attributes without type checks.
        super().__init__(conv)
        self.add = add

    def forward(self, x1, x2):  # type: ignore[override]
        # [FUNCTION DOC] Apply the convolution to x1, add the result to x2 via the add module, and return it. Requires two inputs.
        r"""Applies convolution to x1 and adds the result to x2."""
        return self.add(self[0](x1), x2)


class ConvAddReLU2d(_FusedModule):
    # [CLASS DOC] Sequential container holding the fused submodules (Conv2d, add module, ReLU), in call order. Output is the successive application of each submodule; during quantization it is replaced by the matching fused module.
    r"""This is a sequential container which calls the Conv2d, add, Relu.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, add, relu):
        # [FUNCTION DOC] Build the container from the conv module (kept as the sequential child) and store the add and relu module as attributes without type checks.
        super().__init__(conv)
        self.add = add
        self.relu = relu

    def forward(self, x1, x2):  # type: ignore[override]
        # [FUNCTION DOC] Apply the convolution to x1, add the result to x2 via the add module and applies ReLU, and return it. Requires two inputs.
        r"""Applies convolution to x1, adds the result to x2, and applies ReLU."""
        return self.relu(self.add(self[0](x1), x2))
