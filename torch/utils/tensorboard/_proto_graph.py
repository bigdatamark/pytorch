# [MODULE DOC] Builders for TensorBoard graph protobuf messages (NodeDef, AttrValue,
# TensorShapeProto) used when exporting model graphs. Pure constructors, no import-time effects.
import torch

from collections.abc import Sequence
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto


def attr_value_proto(dtype: object, shape: Sequence[int] | None, s: str | None) -> dict[str, AttrValue]:
    # [FUNCTION DOC] Build the attr mapping for a NodeDef. If s is given, adds key attr holding
    # the utf-8 encoded string; if shape is given, adds key _output_shapes holding a list with
    # one shape proto. Returns the mapping (possibly empty).
    # KNOWN ISSUE: the dtype argument is accepted but never used.
    """Create a dict of objects matching a NodeDef's attr field.

    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/attr_value.proto
    specifically designed for a NodeDef. The values have been reverse engineered from
    standard TensorBoard logged data.
    """
    attr = {}
    if s is not None:
        attr["attr"] = AttrValue(s=s.encode(encoding="utf_8"))
    if shape is not None:
        shapeproto = tensor_shape_proto(shape)
        attr["_output_shapes"] = AttrValue(list=AttrValue.ListValue(shape=[shapeproto]))
    return attr


def tensor_shape_proto(outputsize: Sequence[int]) -> TensorShapeProto:
    # [FUNCTION DOC] Build a TensorShapeProto with one dimension entry per element of outputsize.
    """Create an object matching a tensor_shape field.

    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/tensor_shape.proto .
    """
    return TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in outputsize])


def node_proto(
    name: str,
    op: str = "UnSpecified",
    input: list[str] | str | None = None,
    dtype: torch.dtype | None = None,
    shape: tuple[int, ...] | None = None,
    outputsize: Sequence[int] | None = None,
    attributes: str = "",
) -> NodeDef:
    # [FUNCTION DOC] Build a NodeDef with the utf-8 encoded name, the op, a list of input node
    # names (None becomes empty, a single string is wrapped in a list) and attrs derived from
    # outputsize and attributes via attr_value_proto.
    # KNOWN ISSUE: the shape argument is unused; dtype is passed on but ignored downstream.
    """Create an object matching a NodeDef.

    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/node_def.proto .
    """
    if input is None:
        input = []
    if not isinstance(input, list):
        input = [input]
    return NodeDef(
        name=name.encode(encoding="utf_8"),
        op=op,
        input=input,
        attr=attr_value_proto(dtype, outputsize, attributes),
    )
