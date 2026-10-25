# [MODULE DOC] Schema definition for serialized exported programs: dataclasses and tagged-union classes
# (field values carry integer ids via annotations) describing dtypes, devices, symbolic values, tensor
# metadata, arguments, nodes, graphs, input/output specs, graph signatures, module call info and the
# top-level exported program, plus inference containers (payload metadata, AOT model pickle data, extern
# kernel nodes). Also holds the schema and tree-spec version numbers and mappings from serialized enum
# values to c10 enum member names. Pure data definitions, no import-time side effects beyond class creation.
# NOTE: This is a placeholder for iterating on export serialization schema design.
#       Anything is subject to change and no guarantee is provided at this point.

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Annotated

from torch._export.serde.union import _Union, _union_dataclass


# NOTE: Please update this value if any modifications are made to the schema
SCHEMA_VERSION = (8, 20)
TREESPEC_VERSION = 1


# NOTE: If you updated the schema, please run `scripts/export/update_schema.py`
# to update the auto generated files.
#
# There are also mappings from serialized enum values to c10 enum member names.
# These are used by scripts/export/update_schema.py to generate C++ conversion
# functions for oss_proxy_executor.cpp.
# When adding new enum values, update both the enum AND the mapping.
class ScalarType(IntEnum):
    # [CLASS DOC] Serialized dtype enumeration with stable integer values (note the gap between 13 and 28);
    # each value maps by name to a c10 scalar type via the mapping table below.
    UNKNOWN = 0
    BYTE = 1
    CHAR = 2
    SHORT = 3
    INT = 4
    LONG = 5
    HALF = 6
    FLOAT = 7
    DOUBLE = 8
    COMPLEXHALF = 9
    COMPLEXFLOAT = 10
    COMPLEXDOUBLE = 11
    BOOL = 12
    BFLOAT16 = 13
    UINT16 = 28
    FLOAT8E4M3FN = 29
    FLOAT8E5M2 = 30
    FLOAT8E4M3FNUZ = 31
    FLOAT8E5M2FNUZ = 32
    FLOAT8E8M0FNU = 33
    UINT32 = 34
    UINT64 = 35


class Layout(IntEnum):
    # [CLASS DOC] Serialized tensor layout enumeration (unknown, sparse variants, mkldnn, strided).
    Unknown = 0
    SparseCoo = 1
    SparseCsr = 2
    SparseCsc = 3
    SparseBsr = 4
    SparseBsc = 5
    _mkldnn = 6
    Strided = 7


class MemoryFormat(IntEnum):
    # [CLASS DOC] Serialized memory format enumeration (unknown, contiguous, channels last 2d/3d, preserve).
    Unknown = 0
    ContiguousFormat = 1
    ChannelsLast = 2
    ChannelsLast3d = 3
    PreserveFormat = 4


SCALAR_TYPE_TO_C10: dict[int, str] = {
    ScalarType.BYTE: "Byte",
    ScalarType.CHAR: "Char",
    ScalarType.SHORT: "Short",
    ScalarType.INT: "Int",
    ScalarType.LONG: "Long",
    ScalarType.HALF: "Half",
    ScalarType.FLOAT: "Float",
    ScalarType.DOUBLE: "Double",
    ScalarType.COMPLEXHALF: "ComplexHalf",
    ScalarType.COMPLEXFLOAT: "ComplexFloat",
    ScalarType.COMPLEXDOUBLE: "ComplexDouble",
    ScalarType.BOOL: "Bool",
    ScalarType.BFLOAT16: "BFloat16",
    ScalarType.UINT16: "UInt16",
    ScalarType.FLOAT8E4M3FN: "Float8_e4m3fn",
    ScalarType.FLOAT8E5M2: "Float8_e5m2",
    ScalarType.FLOAT8E4M3FNUZ: "Float8_e4m3fnuz",
    ScalarType.FLOAT8E5M2FNUZ: "Float8_e5m2fnuz",
    ScalarType.FLOAT8E8M0FNU: "Float8_e8m0fnu",
    ScalarType.UINT32: "UInt32",
    ScalarType.UINT64: "UInt64",
}

LAYOUT_TO_C10: dict[int, str] = {
    Layout.SparseCoo: "Sparse",
    Layout.SparseCsr: "SparseCsr",
    Layout.SparseCsc: "SparseCsc",
    Layout.SparseBsr: "SparseBsr",
    Layout.SparseBsc: "SparseBsc",
    Layout._mkldnn: "Mkldnn",
    Layout.Strided: "Strided",
}

MEMORY_FORMAT_TO_C10: dict[int, str] = {
    MemoryFormat.ContiguousFormat: "Contiguous",
    MemoryFormat.ChannelsLast: "ChannelsLast",
    MemoryFormat.ChannelsLast3d: "ChannelsLast3d",
    MemoryFormat.PreserveFormat: "Preserve",
}


@dataclass
class Device:
    # [CLASS DOC] Serialized device: a type string and an optional index (None when unspecified).
    type: Annotated[str, 10]
    index: Annotated[int | None, 20] = None


@_union_dataclass
class SymExprHint(_Union):
    # [CLASS DOC] Tagged union holding the concrete hint value (int, bool or float) of a symbolic expression.
    as_int: Annotated[int, 10]
    as_bool: Annotated[bool, 20]
    as_float: Annotated[float, 30]


# This is for storing the symbolic expressions behind symints/symfloats/symbools
# For example, we can get something like
# SymExpr(expr_str="s0 + s1", hint=SymExprHint(as_int=4)
# if we also have the hint that s0 and s1 are both 2.
@dataclass
class SymExpr:
    # [CLASS DOC] A symbolic expression stored as its string form with an optional concrete hint.
    expr_str: Annotated[str, 10]
    hint: Annotated[SymExprHint | None, 20] = None


@_union_dataclass
class SymInt(_Union):
    # [CLASS DOC] Tagged union: either a symbolic expression or a concrete integer.
    as_expr: Annotated[SymExpr, 10]
    as_int: Annotated[int, 20]


@_union_dataclass
class SymFloat(_Union):
    # [CLASS DOC] Tagged union: either a symbolic expression or a concrete float.
    as_expr: Annotated[SymExpr, 10]
    as_float: Annotated[float, 20]


@_union_dataclass
class SymBool(_Union):
    # [CLASS DOC] Tagged union: either a symbolic expression or a concrete bool.
    as_expr: Annotated[SymExpr, 10]
    as_bool: Annotated[bool, 20]


@dataclass
class TensorMeta:
    # [CLASS DOC] Serialized tensor metadata: dtype, symbolic sizes and strides, requires-grad flag, device,
    # symbolic storage offset and layout.
    dtype: Annotated[ScalarType, 10]
    sizes: Annotated[list[SymInt], 20]
    requires_grad: Annotated[bool, 30]
    device: Annotated[Device, 40]
    strides: Annotated[list[SymInt], 50]
    storage_offset: Annotated[SymInt, 60]
    layout: Annotated[Layout, 70]


# In most cases we will use the "as_name" field to store arguments which are
# SymInts.
# The "as_int" field is used in the case where we have a list containing a mix
# of SymInt and ints (ex. [1, s0, ...]). We will serialize this type of list to
# be List[SymIntArgument] and map the SymInts to the "as_name" field, and ints
# to the "as_int" field.
@_union_dataclass
class SymIntArgument(_Union):
    # [CLASS DOC] Tagged union for a symbolic-int argument: the name of a symbolic value, or a literal int (used when a
    # list mixes symbolic and concrete ints).
    as_name: Annotated[str, 10]
    as_int: Annotated[int, 20]


# In most cases we will use the "as_name" field to store arguments which are
# SymFloats.
# The "as_float" field is used in the case where we have a list containing a mix
# of SymFloat and float (ex. [1.0, s0, ...]). We will serialize this type of list to
# be List[SymFloatArgument] and map the SymFloats to the "as_name" field, and ints
# to the "as_float" field.
@_union_dataclass
class SymFloatArgument(_Union):
    # [CLASS DOC] Tagged union for a symbolic-float argument: a symbolic value name, or a literal float (for mixed lists).
    as_name: Annotated[str, 10]
    as_float: Annotated[float, 20]


# In most cases we will use the "as_name" field to store arguments which are
# SymBools.
# The "as_bool" field is used in the case where we have a list containing a mix
# of SymBool and bools (ex. [True, i0, ...]). We will serialize this type of list to
# be List[SymBoolArgument] and map the SymBools to the "as_name" field, and bools
# to the "as_bool" field.
@_union_dataclass
class SymBoolArgument(_Union):
    # [CLASS DOC] Tagged union for a symbolic-bool argument: a symbolic value name, or a literal bool (for mixed lists).
    as_name: Annotated[str, 10]
    as_bool: Annotated[bool, 20]


@dataclass
class TensorArgument:
    # [CLASS DOC] Reference to a tensor value in the graph by name.
    name: Annotated[str, 10]


@dataclass
class TokenArgument:
    # [CLASS DOC] Reference to an effect-ordering token value in the graph by name.
    name: Annotated[str, 10]


# This is use for storing the contents of a list which contain optional tensors
# (Tensor?[], ex. [Tensor, None, ...]), where the list will be serialized to the
# type List[OptionalTensorArgument], with tensor values serialized to the
# "as_tensor" field, and None values serialized to the "as_none" field.
@_union_dataclass
class OptionalTensorArgument(_Union):
    # [CLASS DOC] Tagged union for an optional tensor element: a tensor reference, or a none marker.
    as_tensor: Annotated[TensorArgument, 20]
    as_none: Annotated[bool, 10]


@dataclass
class GraphArgument:
    # [CLASS DOC] A named subgraph passed as an argument (for example to a higher-order op); holds the nested Graph.
    name: Annotated[str, 10]
    graph: Annotated["Graph", 20]


@dataclass
class CustomObjArgument:
    # [CLASS DOC] Reference to a custom (script) object by name together with its class fully qualified name.
    name: Annotated[str, 10]
    class_fqn: Annotated[str, 20]


@dataclass
class ComplexValue:
    # [CLASS DOC] A complex number literal stored as real and imaginary floats.
    real: Annotated[float, 10]
    imag: Annotated[float, 20]


# This is actually a union type
@_union_dataclass
class Argument(_Union):
    # [CLASS DOC] Tagged union of every kind of value a node input/output can be: none, tensors (single, lists,
    # nested, optional), ints, floats, bools, strings, symbolic values, dtype/layout/memory-format/device
    # literals, subgraphs, custom objects, operator names, complex values, int/float lists of lists, and a
    # mapping of strings to arguments. Exactly one alternative is set per instance.
    as_none: Annotated[bool, 10]
    as_tensor: Annotated[TensorArgument, 20]
    as_tensors: Annotated[list[TensorArgument], 30]
    as_int: Annotated[int, 50]
    as_ints: Annotated[list[int], 70]
    as_float: Annotated[float, 80]
    as_floats: Annotated[list[float], 90]
    as_string: Annotated[str, 100]
    as_strings: Annotated[list[str], 101]
    as_sym_int: Annotated[SymIntArgument, 110]
    as_sym_ints: Annotated[list[SymIntArgument], 120]
    as_scalar_type: Annotated[ScalarType, 130]
    as_memory_format: Annotated[MemoryFormat, 140]
    as_layout: Annotated[Layout, 150]
    as_device: Annotated[Device, 160]
    as_bool: Annotated[bool, 170]
    as_bools: Annotated[list[bool], 180]
    as_sym_bool: Annotated[SymBoolArgument, 182]
    as_sym_bools: Annotated[list[SymBoolArgument], 184]
    as_graph: Annotated[GraphArgument, 200]
    as_optional_tensors: Annotated[list[OptionalTensorArgument], 190]
    as_custom_obj: Annotated[CustomObjArgument, 210]
    as_operator: Annotated[str, 220]
    as_sym_float: Annotated[SymFloatArgument, 230]
    as_sym_floats: Annotated[list[SymFloatArgument], 240]
    as_optional_tensor: Annotated[OptionalTensorArgument, 250]
    as_complex: Annotated[ComplexValue, 260]
    as_nested_tensors: Annotated[list[list[TensorArgument]], 270]
    as_int_lists: Annotated[list[list[int]], 280]
    as_string_to_argument: Annotated[dict[str, "Argument"], 290]
    as_float_lists: Annotated[list[list[float]], 300]


class ArgumentKind(IntEnum):
    # [CLASS DOC] Enumeration of how an argument is passed: unknown, positional or keyword.
    UNKNOWN = 0
    POSITIONAL = 1
    KEYWORD = 2


@dataclass
class NamedArgument:
    # [CLASS DOC] An argument paired with its operator-schema parameter name and an optional passing kind.
    # Argument name from the operator schema
    name: Annotated[str, 10]
    arg: Annotated[Argument, 20]
    kind: Annotated[ArgumentKind | None, 30] = None


@dataclass
class Node:
    # [CLASS DOC] A serialized graph node: target operator name, named inputs, outputs, string metadata, an optional
    # flag for higher-order ops that return a single tensor, and an optional name (optional for loading older models).
    target: Annotated[str, 10]
    inputs: Annotated[list[NamedArgument], 20]
    outputs: Annotated[list[Argument], 30]
    metadata: Annotated[dict[str, str], 40]
    is_hop_single_tensor_return: Annotated[bool | None, 50] = None
    # For BC, default is None so older serialized models without 'name' can be loaded.
    name: Annotated[str | None, 60] = None


@dataclass
class Graph:
    # [CLASS DOC] A serialized graph: input and output arguments, ordered nodes, and tables of tensor, symbolic int, bool
    # and float values and custom objects keyed by name. The single-tensor-return flag marks subgraphs
    # of higher-order ops that return a bare tensor instead of a one-element list.
    inputs: Annotated[list[Argument], 10]
    outputs: Annotated[list[Argument], 20]
    nodes: Annotated[list[Node], 30]
    tensor_values: Annotated[dict[str, TensorMeta], 40]
    sym_int_values: Annotated[dict[str, SymInt], 50]
    sym_bool_values: Annotated[dict[str, SymBool], 60]
    # This is for deserializing the submodule graphs from higher order ops
    # (ex. cond, map) where single tensor returns will just return a single
    # tensor, rather than following export schema and returning a singleton
    # list.
    is_single_tensor_return: Annotated[bool, 70] = False
    custom_obj_values: Annotated[dict[str, CustomObjArgument], 80] = field(
        default_factory=dict
    )
    sym_float_values: Annotated[dict[str, SymFloat], 90] = field(default_factory=dict)


@dataclass
class UserInputSpec:
    # [CLASS DOC] Input spec for a user-supplied input; wraps an argument (only tensors and symbolic ints are expected).
    # Actually, only tensors and SymInts are allowed here
    arg: Annotated[Argument, 10]


@_union_dataclass
class ConstantValue(_Union):
    # [CLASS DOC] Tagged union of a constant input value: none, int, float, string or bool.
    as_none: Annotated[bool, 10]
    as_int: Annotated[int, 20]
    as_float: Annotated[float, 30]
    as_string: Annotated[str, 40]
    as_bool: Annotated[bool, 50]


@dataclass
class InputToConstantInputSpec:
    # [CLASS DOC] Input spec binding a named graph input to a constant value.
    name: Annotated[str, 10]
    value: Annotated[ConstantValue, 20]


@dataclass
class InputToParameterSpec:
    # [CLASS DOC] Input spec binding a tensor input to a module parameter by its qualified name.
    arg: Annotated[TensorArgument, 10]
    parameter_name: Annotated[str, 20]


@dataclass
class InputToBufferSpec:
    # [CLASS DOC] Input spec binding a tensor input to a module buffer by name, with a flag for whether the buffer is persistent.
    arg: Annotated[TensorArgument, 10]
    buffer_name: Annotated[str, 20]
    persistent: Annotated[bool, 30]


@dataclass
class InputToTensorConstantSpec:
    # [CLASS DOC] Input spec binding a tensor input to a lifted tensor constant by name.
    arg: Annotated[TensorArgument, 10]
    tensor_constant_name: Annotated[str, 20]


@dataclass
class InputToCustomObjSpec:
    # [CLASS DOC] Input spec binding a custom-object input to a lifted custom object by name.
    arg: Annotated[CustomObjArgument, 10]
    custom_obj_name: Annotated[str, 20]


@dataclass
class InputTokenSpec:
    # [CLASS DOC] Input spec for an effect-ordering token input.
    arg: Annotated[TokenArgument, 10]


@_union_dataclass
class InputSpec(_Union):
    # [CLASS DOC] Tagged union describing what a graph input is: user input, parameter, buffer, tensor constant, custom
    # object, token, or constant input.
    user_input: Annotated[UserInputSpec, 10]
    parameter: Annotated[InputToParameterSpec, 20]
    buffer: Annotated[InputToBufferSpec, 30]
    tensor_constant: Annotated[InputToTensorConstantSpec, 40]
    custom_obj: Annotated[InputToCustomObjSpec, 50]
    token: Annotated[InputTokenSpec, 70]
    constant_input: Annotated[InputToConstantInputSpec, 60]


@dataclass
class UserOutputSpec:
    # [CLASS DOC] Output spec for an ordinary user-visible output argument.
    arg: Annotated[Argument, 10]


@dataclass
class LossOutputSpec:
    # [CLASS DOC] Output spec marking the tensor that is the training loss.
    arg: Annotated[TensorArgument, 10]


@dataclass
class BufferMutationSpec:
    # [CLASS DOC] Output spec stating that the output tensor is the new value of a named buffer.
    arg: Annotated[TensorArgument, 10]
    buffer_name: Annotated[str, 20]


@dataclass
class ParameterMutationSpec:
    # [CLASS DOC] Output spec stating that the output tensor is the new value of a named parameter.
    arg: Annotated[TensorArgument, 10]
    parameter_name: Annotated[str, 20]


@dataclass
class GradientToParameterSpec:
    # [CLASS DOC] Output spec stating that the output tensor is the gradient of a named parameter.
    arg: Annotated[TensorArgument, 10]
    parameter_name: Annotated[str, 20]


@dataclass
class GradientToUserInputSpec:
    # [CLASS DOC] Output spec stating that the output tensor is the gradient of a named user input.
    arg: Annotated[TensorArgument, 10]
    user_input_name: Annotated[str, 20]


@dataclass
class UserInputMutationSpec:
    # [CLASS DOC] Output spec stating that the output tensor is the new value of a mutated user input.
    arg: Annotated[TensorArgument, 10]
    user_input_name: Annotated[str, 20]


@dataclass
class OutputTokenSpec:
    # [CLASS DOC] Output spec for an effect-ordering token output.
    arg: Annotated[TokenArgument, 10]


@_union_dataclass
class OutputSpec(_Union):
    # [CLASS DOC] Tagged union describing what a graph output is: user output, loss, buffer/parameter/user-input
    # mutation, gradient to parameter or user input, or token.
    user_output: Annotated[UserOutputSpec, 10]
    loss_output: Annotated[LossOutputSpec, 20]
    buffer_mutation: Annotated[BufferMutationSpec, 30]
    gradient_to_parameter: Annotated[GradientToParameterSpec, 40]
    gradient_to_user_input: Annotated[GradientToUserInputSpec, 50]
    user_input_mutation: Annotated[UserInputMutationSpec, 60]
    token: Annotated[OutputTokenSpec, 70]
    parameter_mutation: Annotated[ParameterMutationSpec, 80]


@dataclass
class GraphSignature:
    # [CLASS DOC] The signature of a graph: ordered input specs and output specs, aligned with graph inputs and outputs.
    input_specs: Annotated[list[InputSpec], 10]
    output_specs: Annotated[list[OutputSpec], 20]


@dataclass
class RangeConstraint:
    # [CLASS DOC] Value range of a symbol: optional minimum and maximum (None means unbounded).
    min_val: Annotated[int | None, 10]
    max_val: Annotated[int | None, 20]


@dataclass
class ModuleCallSignature:
    # [CLASS DOC] Calling convention of a submodule: input and output arguments, input/output tree specs stored as
    # strings, and optional original forward argument names used to name placeholders after reloading.
    # KNOWN ISSUE: the comment above the spec fields has load/dump function names swapped relative to the
    # usual meaning (specs are saved with the dump function and read back with the load function).
    inputs: Annotated[list[Argument], 10]
    outputs: Annotated[list[Argument], 20]

    # These are serialized by calling pytree.treespec_loads
    # And deserialized by calling pytree.treespec_dumps
    in_spec: Annotated[str, 30]
    out_spec: Annotated[str, 40]

    # This field is used to prettify the graph placeholders
    # after we Ser/Der and retrace
    forward_arg_names: Annotated[list[str] | None, 50] = None


@dataclass
class ModuleCallEntry:
    # [CLASS DOC] One entry of the module call graph: a module's qualified name and its optional call signature.
    fqn: Annotated[str, 10]
    signature: Annotated[ModuleCallSignature | None, 30] = None


@dataclass
class NamedTupleDef:
    # [CLASS DOC] Field names of a namedtuple type, kept so tree specs can be rebuilt.
    field_names: Annotated[list[str], 10]


@dataclass
class GraphModule:
    # [CLASS DOC] A serialized graph module: the graph, its signature, the module call graph (used to unflatten
    # modules), string metadata, and a mapping from namedtuple type names to field names (kept for compatibility).
    graph: Annotated[Graph, 10]
    signature: Annotated[GraphSignature, 50]
    # This is used for unflattening, by tracking the calling structure of all of
    # the modules in order to unflatten the modules back to the eager calling
    # conventions.
    module_call_graph: Annotated[list[ModuleCallEntry], 60]
    metadata: Annotated[dict[str, str], 40] = field(default_factory=dict)
    # Mapping of namedtuple types to namedtuple field names, used for BC
    treespec_namedtuple_fields: Annotated[dict[str, NamedTupleDef], 70] = field(
        default_factory=dict
    )


# Invariant: Every time a change is made to the schema, one of the versions
#            should be updated.
@dataclass
class SchemaVersion:
    # [CLASS DOC] Schema version as major (bumped on breaking changes) and minor (bumped on compatible changes) numbers.
    major: Annotated[
        int, 10
    ]  # Major version number is bumped every time a breaking change is made.
    minor: Annotated[
        int, 20
    ]  # Minor version number is bumped when a compatible change is made.


@dataclass
class ExportedProgram:
    # [CLASS DOC] Top-level serialized exported program: graph module, opset versions per namespace, range
    # constraints per symbol, schema version, verifier names, the torch version that produced it
    # (defaults to an old-version marker for legacy files) and guard code strings.
    graph_module: Annotated[GraphModule, 10]
    # Key is the opset namespace (ex. aten), and value is the version number
    opset_version: Annotated[dict[str, int], 20]
    range_constraints: Annotated[dict[str, RangeConstraint], 30]
    schema_version: Annotated[SchemaVersion, 60]
    verifiers: Annotated[list[str], 70] = field(default_factory=list)
    torch_version: Annotated[str, 80] = "<=2.4"
    guards_code: Annotated[list[str], 90] = field(default_factory=list)


#########################################################################
# Container types for inference tasks, not being used directly for export.
#########################################################################


# The metadata for payload saved in PT2 archive.
# payload includes params, buffers, tensor constants, and custom objects.
@dataclass
class PayloadMeta:
    # [CLASS DOC] Metadata of one payload (parameter, buffer, tensor constant or custom object) stored in an archive:
    # archive path, parameter flag, whether it is pickled, and optional tensor metadata.
    # the path of the payload in the archive file, e.g. "weight_0"
    path_name: Annotated[str, 10]
    is_param: Annotated[bool, 20]
    # whether the payload is serialized using pickle.
    # Only custom objects and tensor subclasses that are not fake tensors
    # are serialized using pickle.
    use_pickle: Annotated[bool, 30]
    # Custom Objects don't have tensor_meta and will be serialized using pickle
    tensor_meta: Annotated[TensorMeta | None, 40]


# The mapping from payload FQN to its metadata.
@dataclass
class PayloadConfig:
    # [CLASS DOC] Mapping from payload qualified name to its PayloadMeta.
    config: Annotated[dict[str, PayloadMeta], 10]


#
# The structure is used to serialize instances of AOTInductorModel to pass
# them from the publishing pipeline to the predictor.
#
# All new fields should be marked as optional.
#
@dataclass
class AOTInductorModelPickleData:
    # [CLASS DOC] Data to pass an AOTInductor model from a publishing pipeline to a predictor: library file base name,
    # engine input and output names, optional floating point input/output dtype overrides, and an optional CPU flag.
    # New fields must be optional.
    # Base name of an associated .so AOTInductor library. Typically looks like:
    # "abc.so".
    library_basename: Annotated[str, 1]

    # AOTInductor engine input names.
    input_names: Annotated[list[str], 2]

    # AOTInductor engine output names.
    output_names: Annotated[list[str], 3]

    # These fields tell whether floating point inputs/outputs should be converted to
    # a certain type. If None, the dtypes that the AOTInductor engine inferred from the sample
    # inputs are used.
    floating_point_input_dtype: Annotated[int | None, 4] = None
    floating_point_output_dtype: Annotated[int | None, 5] = None

    # Whether AOTInductor runtime is for CPU.
    aot_inductor_model_is_cpu: Annotated[bool | None, 6] = None


@dataclass
class ExternKernelNode:
    # [CLASS DOC] A serialized external-kernel node with a (non-unique) name and the node itself.
    # name is not the unique identifier of the node
    name: Annotated[str, 10]
    node: Annotated[Node, 20]


@dataclass
class ExternKernelNodes:
    # [CLASS DOC] A list of external-kernel nodes.
    nodes: Annotated[list[ExternKernelNode], 10]
