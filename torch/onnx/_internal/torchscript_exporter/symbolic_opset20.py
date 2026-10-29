# mypy: allow-untyped-defs
# [MODULE DOC]
# TorchScript exporter symbolic functions for ONNX opset 20: grid_sampler, affine_grid_generator
# and gelu mapped to the GridSample, AffineGrid and Gelu operators. Exports those three names.
# Import-time side effects: registers the handlers in the global symbolic registry for opset 20.
"""This file exports ONNX ops for opset 20.

Note [ONNX Operators that are added/updated in opset 20]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-20-of-the-default-onnx-operator-set
New operators:
    AffineGrid
    ConstantOfShape
    DFT
    Gelu
    GridSample
    ImageDecoder
    IsInf
    IsNaN
    ReduceMax
    ReduceMin
    RegexFullMatch
    StringConcat
    StringSplit
"""

import functools

import torch.nn.functional as F
from torch import _C
from torch.onnx._internal.torchscript_exporter import (
    jit_utils,
    registration,
    symbolic_helper,
)


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

__all__ = ["_grid_sampler", "_affine_grid_generator", "gelu"]


def convert_grid_sample_mode(mode_s):
    # [FUNCTION DOC]
    # Maps a torch grid sample mode name to the opset 20 GridSample name: bilinear becomes
    # linear, bicubic becomes cubic, anything else (e.g. nearest) is returned unchanged.
    return (
        "linear" if mode_s == "bilinear" else "cubic" if mode_s == "bicubic" else mode_s
    )


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=20)


@_onnx_symbolic("aten::grid_sampler")
@symbolic_helper.parse_args("v", "v", "i", "i", "b")
def _grid_sampler(
    g: jit_utils.GraphContext,
    input: _C.Value,
    grid: _C.Value,
    mode_enum: int,
    padding_mode_enum: int,
    align_corners: bool,
):
    # [FUNCTION DOC]
    # Symbolic for grid_sampler. Looks up the mode and padding mode names from their integer
    # enums via the functional module tables (raises an error for an unknown enum), converts the
    # mode name to the opset 20 spelling, and emits a GridSample op with the align_corners flag
    # as an integer attribute. Returns the graph value.
    mode_s = {v: k for k, v in F.GRID_SAMPLE_INTERPOLATION_MODES.items()}[mode_enum]  # type: ignore[call-arg, index]
    # mode string changes at https://onnx.ai/onnx/operators/text_diff_GridSample_16_20.html
    mode_s = convert_grid_sample_mode(mode_s)
    padding_mode_s = {v: k for k, v in F.GRID_SAMPLE_PADDING_MODES.items()}[  # type: ignore[call-arg, index]
        padding_mode_enum  # type: ignore[index]
    ]
    return g.op(
        "GridSample",
        input,
        grid,
        align_corners_i=int(align_corners),
        mode_s=mode_s,
        padding_mode_s=padding_mode_s,
    )


@_onnx_symbolic("aten::affine_grid_generator")
@symbolic_helper.parse_args("v", "v", "b")
def _affine_grid_generator(
    g: jit_utils.GraphContext,
    theta: _C.Value,
    size: _C.Value,
    align_corners: bool,
):
    # [FUNCTION DOC]
    # Symbolic for affine_grid_generator: emits an AffineGrid op on theta and size with the
    # align_corners flag as an integer attribute and returns the graph value.
    return g.op(
        "AffineGrid",
        theta,
        size,
        align_corners_i=int(align_corners),
    )


@_onnx_symbolic("aten::gelu")
@symbolic_helper.parse_args("v", "s")
def gelu(g: jit_utils.GraphContext, self: _C.Value, approximate: str = "none"):
    # [FUNCTION DOC]
    # Symbolic for gelu: emits a Gelu op on the input with the approximate mode string
    # attribute (none or tanh) and returns the graph value.
    return g.op("Gelu", self, approximate_s=approximate)
