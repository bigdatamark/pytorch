# [MODULE DOC]
# Generates C++ for lazy tensor backends: IR node class declarations (generic and
# TorchScript flavored), native function definitions that build IR nodes and wrap results
# as ATen tensors, shape inference declarations, and non-native IR nodes. Also holds helper
# functions rendering node constructor arguments and eager fallback code. No import-time
# side effects.
from __future__ import annotations

import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any

import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import (
    getValueT,
    isValueType,
    LazyArgument,
    LazyIrProperties,
    LazyIrSchema,
    tensorListValueT,
)
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    deviceT,
    DispatcherSignature,
    kernel_signature,
    NativeSignature,
    OptionalCType,
    VectorCType,
)
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import (
    Argument,
    BackendIndex,
    BackendMetadata,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    NativeFunctionsGroup,
)


def node_ctor_arg_rvalue_string(arg: LazyArgument) -> str:
    # [FUNCTION DOC]
    # Returns a C++ expression passing one argument to a lazy node constructor. Value-typed
    # arguments become IR values (wrapped scalar node, tensor list, symint value, or the
    # lazy tensor's IR value, with optional forms guarded); other arguments are passed as
    # is or converted to vectors / optional vectors / symint arrays.
    # Raises an error for a value type that is neither a base nor an optional C type.
    """
    Given a LazyArgument,
    generate a c++ string for materializing an rvalue of that arg for passing into
    a lazy Node constructor.
    """

    # TODO: Matching on CType seems wrong; should be matching on Type
    if isValueType(arg.lazy_type):
        if isinstance(arg.lazy_type, BaseCType):
            if arg.is_wrapped_scalar:
                return f"node_{arg.name}"
            elif arg.lazy_type.type is tensorListValueT:
                return f"lazy_{arg.name}_tensorlist"
            elif arg.is_symint_or_list:
                return f"GetSymIntValue({arg.name})"
            return f"lazy_{arg.name}->GetIrValue()"
        elif isinstance(arg.lazy_type, OptionalCType):
            if arg.is_symint_or_list:
                # TODO: I don't understand when you should put lazy_ in the name
                # or not
                return f"{arg.name} ? std::make_optional(GetSymIntValue(*{arg.name})) : ::std::nullopt"
            elif arg.is_wrapped_scalar:
                return f"node_{arg.name}"
            return (
                f"lazy_{arg.name} ? "
                f"std::make_optional(lazy_{arg.name}->GetIrValue()) : "
                "::std::nullopt"
            )
        else:
            raise AssertionError(
                f"TODO not sure if there are other valid types to handle here ({arg.lazy_type})"
            )
    else:
        # NB: this is here because right now we aren't treating SymInt[] as a
        # value type; when we do this needs to move above
        # NB: we cannot test arg.lazy_type as we've already specified it is an
        # int64_t and so we cannot distinguish between SymInt and int64_t
        if isinstance(arg.orig_type, ListType) and arg.orig_type.elem == BaseType(
            BaseTy.SymInt
        ):
            if arg.symint:
                return f"GetSymIntArrayRefValue({arg.name})"
            else:
                return f"std::vector<int64_t>({arg.name}.begin(), {arg.name}.end())"
        elif isinstance(arg.lazy_type, VectorCType) and isinstance(
            arg.lazy_type.elem, BaseCType
        ):
            return f"std::vector<{arg.lazy_type.elem.type}>({arg.name}.begin(), {arg.name}.end())"
        elif (
            isinstance(arg.lazy_type, OptionalCType)
            and isinstance(arg.lazy_type.elem, VectorCType)
            and isinstance(arg.lazy_type.elem.elem, BaseCType)
        ):
            return f"torch::lazy::ToOptionalVector<{arg.lazy_type.elem.elem.type}>({arg.name})"
        else:
            return f"{arg.name}"


def node_ctor_inputs(schema: LazyIrSchema) -> str:
    # [FUNCTION DOC]
    # Returns the comma-separated constructor argument expressions for all filtered
    # arguments of the schema, in order.
    """
    Produce a formatted string with the arguments as passed into the constructor of a node class.
    """
    node_ctor_values = [
        node_ctor_arg_rvalue_string(arg) for arg in schema.filtered_args()
    ]
    return ", ".join(node_ctor_values)


def gen_fallback_code(
    schema: LazyIrSchema,
    sig: DispatcherSignature | NativeSignature,
    overload_name: str,
) -> str:
    # [FUNCTION DOC]
    # Returns a C++ conditional that, when the forced-eager-fallback check for the op
    # symbol is true, returns the result of the eager fallback call with arguments
    # translated from the given signature to the dispatcher signature. The overload name,
    # if non-empty, selects the two-part ATen op macro.
    """
    Generate code that falls back to eager conditioned on a predicate
    """
    dispatcher_sig = DispatcherSignature.from_schema(schema.func)
    exprs = translate(sig.arguments(), dispatcher_sig.arguments())
    fallback_args = ",\n                ".join([a.expr for a in exprs])
    if len(overload_name):
        aten_op_str = f"ATEN_OP2({schema.aten_name}, {overload_name})"
    else:
        aten_op_str = f"ATEN_OP({schema.aten_name})"
    return f"""
        if (force_eager_fallback({aten_symbol(schema)})) {{
            return at::native::call_fallback_fn_symint<&ltc_eager_fallback, {aten_op_str}>::call(
                {fallback_args}
            );
        }}
"""


def aten_symbol(schema: LazyIrSchema) -> str:
    # [FUNCTION DOC]
    # Returns a C++ expression for the ATen symbol of the op: a qualified-string symbol for
    # names missing from interned strings, the name itself if already prefixed with at::,
    # otherwise the name under the at::aten namespace.
    missing_interned_strings = {
        "sigmoid_backward",
    }
    if schema.aten_name in missing_interned_strings:
        return f'c10::Symbol::fromQualString("aten::{schema.aten_name}")'

    if not schema.aten_name.startswith("at::"):
        return f"at::aten::{schema.aten_name}"
    else:
        return schema.aten_name


# converts  all tensor-like arguments to meta tensors. Returns:
# (1) a string containing all of the logic that does the conversions.
# (2) a context, to be used by translate(), with all of the relevant bindings.
def convert_to_meta_tensors(sig: DispatcherSignature) -> tuple[str, list[Binding]]:
    # [FUNCTION DOC]
    # Returns the C++ statements converting every tensor-like argument to a meta tensor
    # (named with a _meta suffix), and the binding context in which those arguments are
    # replaced by the converted names while other arguments are unchanged.
    context: list[Binding] = []
    unwrapped_tensor_args: list[str] = []
    for arg in sig.arguments():
        if isinstance(arg.argument, Argument) and arg.argument.type.is_tensor_like():
            unwrapped_name = f"{arg.name}_meta"
            unwrapped_tensor_args.append(
                f"auto {unwrapped_name} = to_meta({arg.name});"
            )
            context.append(arg.with_name(unwrapped_name))
        else:
            context.append(arg)
    unwrap_tensor_args_str = "\n        ".join(unwrapped_tensor_args)
    return unwrap_tensor_args_str, context


@dataclass(frozen=True)
class GenLazyIR(ABC):
    # [CLASS DOC]
    # Immutable generator of lazy IR node class declarations. State: the backend index,
    # backend name, the C++ node base class name, and whether lazy shape objects are used.
    # Hook methods (lowering, create, reuse check) return empty or trivial text here and
    # are overridden by backend subclasses; gen renders the full node class text.
    backend_index: BackendIndex
    backend_name: str
    node_base: str
    use_lazy_shape: bool

    @method_with_native_function
    def __call__(self, f: NativeFunctionsGroup | NativeFunction) -> list[str]:
        # [FUNCTION DOC]
        # Builds the lazy IR schema from the functional variant (symint-aware according to
        # the kernel metadata) and returns the generated node class text list from gen.
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        metadata = self.backend_index.get_kernel(
            f.functional if isinstance(f, NativeFunctionsGroup) else f
        )
        schema = LazyIrSchema(
            func, symint=metadata is not None and metadata.supports_symint()
        )
        return self.gen(schema)

    # there is no lowering functionality generated unless this IR base class is subclassed and
    # implemented as a backend-specific node
    def lowering_function(self, schema: LazyIrSchema) -> str:
        # [FUNCTION DOC] Hook: returns the lowering method text; the base returns empty text.
        return ""

    def create_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        # [FUNCTION DOC] Hook: returns the node Create method text; the base returns empty.
        return ""

    def can_be_reused_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        # [FUNCTION DOC] Hook: returns a CanBeReused method that always returns false.
        return f"""bool CanBeReused({node_ctor_args}) const {{
    return false;
    }}"""

    def node_base_ctor_call(self, schema: LazyIrSchema) -> str:
        # [FUNCTION DOC]
        # Returns the C++ call of the node base constructor: op kind, operand list from
        # value arguments (optionals use a null default), a shape argument chosen by the
        # shape property (precomputed, computed, or cached lambda; else none), output
        # count, and a hash of the scalar arguments.
        # Raises an error for a value argument whose type is not base, vector or optional.
        value_args = schema.filtered_args(values=True, scalars=False)
        # backends can customize the way the node base class constructor is called,
        # as long as all of its arguments can be generated from information available from the schema
        base_ctor_value_args_list = []
        for arg in value_args:
            if isinstance(arg.lazy_type, (BaseCType, VectorCType)):
                base_ctor_value_args_list.append(f"{arg.name}")
            elif isinstance(arg.lazy_type, OptionalCType):
                base_ctor_value_args_list.append(f"{arg.name}.value_or(kNullValue)")
            else:
                raise AssertionError(
                    f"Unsupported type ({arg.lazy_type}) - add support if necessary"
                )
        base_ctor_value_args = ", ".join(base_ctor_value_args_list)

        scalar_args = schema.filtered_args(values=False, scalars=True)

        # Shape construction.
        # Conditionally build shape depending on specified shape property
        if schema.properties.ShapePrecompute:
            shape_ctor_arg = "std::move(shapes),"
        elif schema.properties.ShapeCompute:
            shape_args = [a.name for a in value_args]
            shape_args.extend(a.name for a in scalar_args)
            shape_ctor_arg = f"compute_shape_{schema.name}({', '.join(shape_args)}),"
        elif schema.properties.ShapeCache:
            shape_args = [f"operand({i})" for i in range(len(value_args))]
            shape_args.extend(a.name for a in scalar_args)
            shape_ctor_arg = f"[&](){{ return compute_shape_{schema.name}({', '.join(shape_args)})[0]; }},"
        else:
            shape_ctor_arg = ""

        scalar_hashes = ", ".join(f"{a.name}" for a in scalar_args)

        return f"""{self.node_base}(
              {schema.node_name}::ClassOpKind(),
              OpList{{{base_ctor_value_args}}},
              {shape_ctor_arg}
              /* num_outputs */ {len(schema.returns)},
              torch::lazy::MHash({scalar_hashes}))"""

    def gen(self, schema: LazyIrSchema) -> list[str]:
        # [FUNCTION DOC]
        # Returns a one-element list with the C++ class text for the node: op kind static
        # method, constructor (with an extra shapes parameter when shape is precomputed),
        # ToString, create, reuse and lowering hooks, scalar member declarations (string views
        # stored as strings), and bit flags recording presence of optional value arguments.
        opkind = schema.opkind or aten_symbol(schema)

        # for now, we just want one IR class decl and soon after also the method defs
        # and we use the functional version not out/inplace.
        all_args = schema.filtered_args()
        scalar_args = schema.filtered_args(values=False, scalars=True)

        ctor_args = [f"const {i.lazy_type.cpp_type()}& {i.name}" for i in all_args]
        reuse_ctor_args = ", ".join(ctor_args)
        if self.use_lazy_shape and schema.properties.ShapePrecompute:
            ctor_args.append("std::vector<torch::lazy::Shape>&& shapes")
        node_ctor_args = ", ".join(ctor_args)

        scalar_initializers = ",\n        ".join(
            [
                # This code is just special casing the mapping from string_view -> strings
                f"{a.name}({a.name}.has_value() ? ::std::make_optional(std::string(*{a.name})) : ::std::nullopt)"
                if a.lazy_type.cpp_type() == "::std::optional<c10::string_view>"
                else f"{a.name}({a.name})"
                for a in scalar_args
            ]
        )
        if len(scalar_initializers):
            scalar_initializers = f",\n        {scalar_initializers}"
        scalar_decls = "\n  ".join(
            [
                f"std::string {a.name};"
                if a.lazy_type.cpp_type() == "c10::string_view"
                else f"::std::optional<std::string> {a.name};"
                if a.lazy_type.cpp_type() == "::std::optional<c10::string_view>"
                else f"{a.lazy_type.cpp_type()} {a.name};"
                for a in scalar_args
            ]
        )
        optional_values = [
            arg.name
            for arg in schema.filtered_args(values=True, scalars=False)
            if isinstance(arg.lazy_type, OptionalCType)
        ]
        has_optional_decls = "\n  ".join(
            [f"bool has_{value}: 1;" for value in optional_values]
        )
        has_optional_defs = "\n    ".join(
            [f"has_{value} = !!{value};" for value in optional_values]
        )
        members_to_string = []
        for arg in scalar_args:
            if isinstance(arg.lazy_type, OptionalCType):
                value = f"{arg.name}.value()"
                if arg.is_generator:
                    value = '"torch.Generator()"'
                members_to_string.append(
                    f"""if ({arg.name}.has_value()) {{
      ss << ", {arg.name}=" << {value};
    }} else {{
      ss << ", {arg.name}=null";
    }}"""
                )
            else:
                members_to_string.append(f'ss << ", {arg.name}=" << {arg.name};')
        members_to_string_str = "\n    ".join(members_to_string)

        return [
            f"""\
class {schema.node_name} : public {self.node_base} {{
 public:
  static torch::lazy::OpKind ClassOpKind() {{
    return torch::lazy::OpKind({opkind});
  }}

  {schema.node_name}({node_ctor_args})
      : {self.node_base_ctor_call(schema)}{scalar_initializers}
  {{
    {has_optional_defs}
  }}

  std::string ToString() const override {{
    std::stringstream ss;
    ss << {self.node_base}::ToString();
    {members_to_string_str}
    return std::move(ss).str();
  }}

  {self.create_function(schema, reuse_ctor_args)}

  {self.can_be_reused_function(schema, reuse_ctor_args)}

  {self.lowering_function(schema)}

  {scalar_decls}
  {has_optional_decls}

}};

""",
        ]


@dataclass(frozen=True)
class GenTSLazyIR(GenLazyIR):
    # [CLASS DOC]
    # TorchScript backend flavor of GenLazyIR: implements the lowering, create and reuse
    # hooks according to the schema's properties (declaration only, full definition, or none).
    def lowering_function(self, schema: LazyIrSchema) -> str:
        # [FUNCTION DOC]
        # Returns the Lower override: only a declaration if LowerDeclOnly, a full definition
        # with the TorchScript lowering body if Lower, otherwise empty text.
        signature = """
  torch::lazy::TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      torch::lazy::TSLoweringContext* loctx) const override"""

        if schema.properties.LowerDeclOnly:
            return f"{signature};"
        elif schema.properties.Lower:
            return f"""{signature} {{
    {ts_lowering_body(schema)}
  }}
            """
        else:
            return ""

    def create_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        # [FUNCTION DOC]
        # Returns the static Create method: declaration only if CreateFnDeclOnly, empty if
        # CreateFn is unset, otherwise a definition reusing or making the node.
        # KNOWN ISSUE: the generated body passes an identifier named data that is not a
        # parameter of the generated signature, so it only compiles if one is in scope.
        signature = f"static NodePtr Create({node_ctor_args})"
        if schema.properties.CreateFnDeclOnly:
            return f"{signature};"
        elif not schema.properties.CreateFn:
            return ""
        return f"""{signature} {{
    return ReuseOrMakeNode<{schema.node_name}>(data);
  }}"""

    def can_be_reused_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        # [FUNCTION DOC]
        # Returns the CanBeReused method: declaration only if CanBeReusedDeclOnly, empty if
        # CanBeReused is unset, otherwise a definition comparing each operand with the value
        # arguments (optionals via null default) and each scalar member with its argument
        # (optionals compared by presence then value), all combined by logical and.
        signature = f"bool CanBeReused({node_ctor_args}) const"
        if schema.properties.CanBeReusedDeclOnly:
            return f"{signature};"
        elif not schema.properties.CanBeReused:
            return ""
        value_comparison = []
        for arg in itertools.chain(schema.positional_values, schema.keyword_values):
            if isinstance(arg.lazy_type, OptionalCType):
                value_comparison.append(
                    f"nullable_operand(i++) == {arg.name}.value_or(kNullValue)"
                )
            else:
                value_comparison.append(f"operand(i++) == {arg.name}")
        for arg in itertools.chain(schema.positional_scalars, schema.keyword_scalars):
            if isinstance(arg.lazy_type, OptionalCType):
                value_comparison.append(
                    f"((!this->{arg.name}&&!{arg.name}) || (this->{arg.name}&&{arg.name} && *(this->{arg.name}) == *{arg.name}))"
                )
            else:
                value_comparison.append(f"this->{arg.name} == {arg.name}")
        value_comparison_str = " &&\n        ".join(value_comparison)

        return f"""{signature} {{
    size_t i = 0;
    return ({value_comparison_str});
  }}"""


@dataclass(frozen=True)
class GenLazyNativeFuncDefinition:
    # [CLASS DOC]
    # Immutable generator of lazy backend native function definitions. State holds the
    # backend index and backend-specific C++ names (class, namespace, helpers for tensor
    # lists, wrapped numbers, tensor creation, counters, device lookup, tensor pointer
    # type). Each call renders one kernel body: fallback, metrics, device, lazy tensor
    # setup, IR node build, and returning the wrapped result.
    class_method_name: str
    backend_index: BackendIndex
    tensor_class: str
    gen_forced_fallback_code: bool
    backend_namespace: str
    get_tensorlist: str
    get_tensor_or_wrap_number: str
    try_get_tensor: str
    metrics_counter: str
    create_tensor: str
    create_from_first_tensor: bool
    create_aten_from_ltc_tensor: str
    tuple_aten_from_ltc_tensors: str
    lazy_tensor_ptr: str
    get_device_fn: str

    def lazy_tensor_decls(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        # [FUNCTION DOC]
        # Returns C++ statements declaring lazy tensor variables (or scalar IR nodes, or
        # tensor list wrappers) for each value argument, one per line. Symint arguments are
        # skipped. Raises an error for an unexpected optional element type or value type.
        value_args = schema.filtered_args(values=True, scalars=False)
        # Generates lazy_{name} variables for LazyTensors wrapping input tensors
        lazy_tensor_decls: list[str] = []
        for arg in value_args:
            if arg.is_wrapped_scalar:
                if isinstance(arg.lazy_type, OptionalCType):
                    lazy_tensor_decls.append(
                        f"""auto node_{arg.name} = {arg.name} ?
                std::make_optional(torch::lazy::LazyGraphExecutor::Get()->
                    GetIrValueForScalarFromCodegen(*{arg.name}, *common_device)):
                ::std::nullopt;"""
                    )
                else:
                    lazy_tensor_decls.append(
                        f"""auto node_{arg.name} = torch::lazy::LazyGraphExecutor::Get()->
                            GetIrValueForScalarFromCodegen({arg.name}, *common_device);"""
                    )
            elif arg.is_symint_or_list:
                continue  # values are extracted in isValueType
            elif isinstance(arg.lazy_type, BaseCType):
                if arg.lazy_type.type is tensorListValueT:
                    lazy_tensor_decls.append(
                        f"auto lazy_{arg.name}_tensorlist = "
                        f"{self.backend_namespace}::{self.get_tensorlist}({arg.name});"
                    )
                else:
                    lazy_tensor_decls.append(
                        f"{self.lazy_tensor_ptr} lazy_{arg.name} = "
                        f"{self.backend_namespace}::{self.get_tensor_or_wrap_number}({arg.name}, *common_device);"
                    )
            elif isinstance(arg.lazy_type, OptionalCType):
                if arg.lazy_type.elem != BaseCType(getValueT()):
                    raise AssertionError(
                        f"Expected OptionalCType elem to be {BaseCType(getValueT())}, "
                        f"got {arg.lazy_type.elem}"
                    )
                # TODO(alanwaketan): Maybe we want to apply GetLtcTensorOrCreateForWrappedNumber here, but hold it
                # until we encounter a real world example.
                lazy_tensor_decls.append(
                    f"{self.lazy_tensor_ptr} lazy_{arg.name} = "
                    f"{self.backend_namespace}::{self.try_get_tensor}({arg.name}.value_or(at::Tensor()));"
                )
            else:
                raise AssertionError(
                    f"TODO not sure if there are other valid types to handle here ({arg.lazy_type})"
                )
        return ("\n        ").join(lazy_tensor_decls)

    def force_eager_fallback(
        self,
        func: NativeFunction,
        schema: LazyIrSchema,
        metadata: BackendMetadata,
        sig: DispatcherSignature | NativeSignature,
    ) -> str:
        # [FUNCTION DOC]
        # Returns the eager fallback conditional text when forced fallback code generation
        # is enabled, otherwise empty text. The metadata argument is unused.
        if self.gen_forced_fallback_code:
            return gen_fallback_code(
                schema, sig, overload_name=func.func.name.overload_name
            )
        return ""

    def metrics(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        # [FUNCTION DOC] Returns the metrics counter statement configured for the backend.
        return f"{self.metrics_counter};"

    def get_device(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        # [FUNCTION DOC]
        # Returns C++ computing the common device from the non-wrapped value arguments and
        # optional device scalar arguments, followed by an internal assertion that it exists.
        # Raises an error if there are neither value arguments nor optional device arguments.
        value_args = schema.filtered_args(values=True, scalars=False)
        scalar_args = schema.filtered_args(values=False, scalars=True)
        value_types_names = [f"{a.name}" for a in value_args if not a.is_wrapped_scalar]
        optional_device = OptionalCType(BaseCType(deviceT))
        optional_devices = [
            a.name for a in scalar_args if a.lazy_type == optional_device
        ]
        if len(value_types_names) == 0 and len(optional_devices) == 0:
            raise AssertionError("Expected at least one Value or Device type")
        get_device_str = (
            f"{self.get_device_fn}({', '.join(value_types_names + optional_devices)})"
        )
        return f"""auto common_device = {get_device_str};
        TORCH_INTERNAL_ASSERT(common_device);
        """

    def shape_inference(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        # [FUNCTION DOC]
        # Returns C++ computing output shapes. Structured and view_copy ops convert tensor
        # arguments to meta tensors and call the meta (or composite non-functional) function
        # to derive shape and dtype per output; other ops call a user-provided compute_shape
        # function. Then asserts the shape count and optionally applies symbolic shapes.
        # Raises an error if kernel metadata is missing or a view_copy op lacks the required
        # composite kernel.
        metadata = self.backend_index.get_kernel(func)
        if metadata is None:
            raise AssertionError(f"No kernel metadata found for {func.func.name}")
        all_args = schema.filtered_args()
        returns_length = len(schema.returns)
        # call the meta kernel if it exists, to compute output shape/dtype for our IR
        # Note [Generated LTC Shape Functions]
        # LTC uses meta tensors from core to do shape inference when possible, and otherwise
        # we generate a shape function declaration that needs to be manually implemented.
        # How do we detect which ops are eligible to use meta tensors?
        # In general we should be able to use meta tensors not just on structured operators,
        # but also on composite operators that are implemented in terms of structured kernels.
        # We don't currently have a way of knowing at codegen time which ops are implemented that way.
        # This is the case for all view and view_copy operators however, so we're going to
        # use them specifically for all of the view_copy ops (instead of manually writing shape rules for all of them).
        is_view_copy_op = "view_copy" in func.tags
        is_structured = func.structured or func.structured_delegate is not None
        if is_structured or is_view_copy_op:
            meta_out = """
std::vector<torch::lazy::Shape> shapes{torch::lazy::Shape(out_meta.scalar_type(), out_meta.sizes().vec())};"""
            if returns_length > 1:

                def this_shape(i: int) -> str:
                    # [FUNCTION DOC] HELPER: Returns C++ building the shape of output i from
                    # the tuple of meta outputs.
                    return f"torch::lazy::Shape(std::get<{i}>(out_meta).scalar_type(), std::get<{i}>(out_meta).sizes().vec())"

                shapes_str = ",".join([this_shape(i) for i in range(returns_length)])
                meta_out = "std::vector<torch::lazy::Shape> shapes{" + shapes_str + "};"

            # Convert tensor args to the meta device and call it.
            # (We can't pass in the input tensors directly, because they are "functional wrappers".
            # If any of the meta kernels call a tensor op and redispatch, we don't want to hit the functionalize kernels.)
            # Even at::meta:: functions might redispatch, e.g. if they call into view ops.
            dispatcher_sig = DispatcherSignature.from_schema(func.func)
            meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
            meta_call_args = [
                e.expr
                for e in translate(
                    meta_call_ctx, dispatcher_sig.arguments(), method=False
                )
            ]
            if is_view_copy_op:
                # view_copy ops always have a CompositeExplicitAutogradNonFunctional kernel
                if not func.has_composite_explicit_autograd_non_functional_kernel:
                    raise AssertionError(
                        f"view_copy op {func.func.name} must have "
                        "CompositeExplicitAutogradNonFunctional kernel"
                    )
                dispatch_ns = "compositeexplicitautogradnonfunctional"
            else:
                dispatch_ns = "meta"
            aten_name = schema.aten_name
            # TODO: this is trolling
            if func.func.has_symint() and metadata.supports_symint():
                aten_name += "_symint"
            shape_str = f"""\
        {meta_conversion_str}
        auto out_meta = at::{dispatch_ns}::{aten_name}({", ".join(meta_call_args)});
        {meta_out}"""
        else:
            shape_sig = ComputeShapeSignature(
                metadata.kernel, func, symint=metadata.supports_symint()
            )
            shape_str = f"""
            auto shapes = {shape_sig.shape_call};"""

        shape_str += f"""
            TORCH_INTERNAL_ASSERT(shapes.size() == {returns_length});"""

        # Calculating which dimensions are symbolic
        func_schema_str = "aten::" + str(func.func)
        shape_str += f"""
            if(torch::lazy::symbolicShapeEnabled()){{
                std::vector<torch::jit::IValue> inputs = {{ {", ".join(str(a.name) for a in all_args)} }};
                const char* schema_str = "{func_schema_str}";
                applySymbolicShapesOnLT(schema_str, inputs, shapes);
            }}
        """
        return shape_str

    def build_ir_node(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        # [FUNCTION DOC]
        # Returns C++ that tries to reuse a cached node and, if none, computes shapes,
        # makes a new node with them and caches it.
        node_ctor_input_str = node_ctor_inputs(schema)
        return f"""torch::lazy::NodePtr node = torch::lazy::ReuseNode<{schema.node_name}>({node_ctor_input_str});
        if (!node) {{
            {self.shape_inference(func, schema)}
            node = torch::lazy::MakeNode<{schema.node_name}>({node_ctor_input_str}, std::move(shapes));
            CacheNode(node);
        }}
        """

    def create_lazy_tensor(self, first_tensor_name: str | None = None) -> str:
        # [FUNCTION DOC]
        # Returns the C++ callee prefix for creating a lazy tensor: a method on the first
        # tensor if the backend creates from the first tensor, else a namespaced function.
        # Raises an error if the first-tensor form is needed but no name is given.
        # xla uses an instance method for tensor creation, for the time being
        if self.create_from_first_tensor:
            # TODO(whc) remove this if XLA switches to using static method for creation
            if first_tensor_name is None:
                raise AssertionError("Requires first tensor to create lazy tensor")
            return f"{first_tensor_name}.{self.create_tensor}"
        return f"{self.backend_namespace}::{self.create_tensor}"

    def return_aten_tensor(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        # [FUNCTION DOC]
        # Returns C++ wrapping the node as the op result: a single ATen tensor from a new lazy
        # tensor, a tuple built from one lazy tensor per output, or, for in-place and out
        # variants, setting the IR value on the first tensor and returning it.
        # Raises an error for multiple outputs without tensor arguments, or for an in-place
        # or out variant with a number of outputs other than one.
        returns_length = len(schema.returns)
        value_args = schema.filtered_args(values=True, scalars=False)
        value_types_names = [f"{a.name}" for a in value_args if not a.is_wrapped_scalar]
        first_tensor_name = value_types_names[0] if len(value_types_names) > 0 else None
        bridge_str = f"""auto result = {self.create_aten_from_ltc_tensor}(
                {self.create_lazy_tensor(first_tensor_name)}(std::move(node), *common_device));"""

        if returns_length > 1:
            if len(value_types_names) == 0:
                raise AssertionError(
                    "Code below assumes there is at least one tensor arg"
                )
            bridge_str = f"""std::vector<{self.lazy_tensor_ptr}> lazy_tensors;
        for (int i = 0; i < {returns_length}; i++) {{
            lazy_tensors.push_back({self.create_lazy_tensor(first_tensor_name)}({getValueT()}(node, i), *common_device));
        }}
        auto result = {self.tuple_aten_from_ltc_tensors}<{returns_length}>(lazy_tensors);"""

        if schema.name.name.inplace or func.func.is_out_fn():
            if returns_length != 1:
                raise AssertionError(
                    "We assumed there was no such case where an op is an in-place variant "
                    f"and has tuple outputs, but got tuple of len {returns_length}."
                )
            bridge_str = f"""lazy_{first_tensor_name}->SetInPlaceIrValue(node);
        auto& result = {first_tensor_name};"""

        bridge_str += """
        return result;"""
        return bridge_str

    @method_with_native_function
    def __call__(self, func: NativeFunction) -> list[str]:
        # [FUNCTION DOC]
        # Returns a one-element list with the full C++ definition of the backend kernel for
        # the function, composing fallback, metrics, device, lazy tensor, IR node and return
        # sections. Raises an error if no kernel metadata exists for the function.
        sig = kernel_signature(func, self.backend_index)
        metadata = self.backend_index.get_kernel(func)
        if metadata is None:
            raise AssertionError(f"No kernel metadata found for {func.func.name}")
        schema = LazyIrSchema(func.func, symint=metadata.supports_symint())
        return [
            f"""\
    {sig.decl(name=f"{self.class_method_name}::{metadata.kernel}")} {{
        {self.force_eager_fallback(func, schema, metadata, sig)}
        {self.metrics(func, schema)}
        {self.get_device(func, schema)}
        {self.lazy_tensor_decls(func, schema)}
        {self.build_ir_node(func, schema)}
        {self.return_aten_tensor(func, schema)}
    }}\n
    """
        ]


class ComputeShapeSignature:
    # [CLASS DOC]
    # Builds the declaration and call text of the compute_shape function for a kernel,
    # from the dispatcher argument declarations and the schema's filtered call arguments.
    # State is stored in name-mangled private attributes set at construction.
    """
    Here we use the base name as the suffix of the signature to avoid generating for in-place variants.
    """

    def __init__(self, kernel_name: str, f: NativeFunction, *, symint: bool) -> None:
        # [FUNCTION DOC]
        # Stores the kernel name, the lazy schema, the comma-separated dispatcher parameter
        # declarations and the comma-separated call arguments (generator included).
        self.__schema = LazyIrSchema(f.func, symint=symint)
        self.__dispatch_args = ", ".join(
            [a.decl() for a in dispatcher.arguments(f.func, symint=symint)]
        )
        self.__call_args = ", ".join(
            [f"{arg.name}" for arg in self.__schema.filtered_args(generator=True)]
        )
        self.__kernel_name = kernel_name

    def __decl_suffix(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the kernel name with its parameter declarations.
        return f"{self.__kernel_name}({self.__dispatch_args})"

    def __call_suffix(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the kernel name with its call arguments.
        return f"{self.__kernel_name}({self.__call_args})"

    @property
    def shape_decl(self) -> str:
        # [FUNCTION DOC] Property: the exported C++ declaration of the compute_shape function.
        return f"TORCH_API std::vector<torch::lazy::Shape> compute_shape_{self.__decl_suffix()}"

    @property
    def shape_call(self) -> str:
        # [FUNCTION DOC] Property: the C++ call expression of the lazy compute_shape function.
        return f"torch::lazy::compute_shape_{self.__call_suffix()}"


@dataclass(frozen=True)
class GenLazyShapeInferenceDefinition:
    # [CLASS DOC]
    # Immutable generator of shape inference function declarations for lazy backends.
    # State: the backend index and the tensor class name (unused by the generator).
    backend_index: BackendIndex
    tensor_class: str

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> list[str]:
        # [FUNCTION DOC]
        # Returns a one-element list with the compute_shape declaration for the function, or
        # an empty list for structured and view_copy ops (shapes come from meta kernels).
        # Raises an error if no kernel metadata exists.
        metadata = self.backend_index.get_kernel(f)
        if metadata is None:
            raise AssertionError(f"No kernel metadata found for {f.func.name}")

        # See Note [Generated LTC Shape Functions]
        is_view_copy_op = "view_copy" in f.tags
        is_structured = f.structured or f.structured_delegate is not None
        if is_structured or is_view_copy_op:
            return []
        else:
            shape_sig = ComputeShapeSignature(
                metadata.kernel, f, symint=metadata.supports_symint()
            )
            return ["\n".join([f"{shape_sig.shape_decl};"])]


def generate_non_native_lazy_ir_nodes(
    non_native: list[dict[str, Any]], gen_lazy_ir: GenLazyIR
) -> list[str]:
    # [FUNCTION DOC]
    # Returns node class text for each non-native op description (mapping with a schema
    # string, optional properties list and optional op kind). Default properties are shape
    # cache, can-be-reused and lower-declaration-only, plus any listed ones; symint is
    # assumed. Uses the given generator's gen method.
    """Generate the non-native lazy IR node classes"""
    nodes = []
    for op in non_native:
        # Set default properties for Non-Native IRs
        properties = LazyIrProperties("ShapeCache", "CanBeReused", "LowerDeclOnly")
        for p in op.get("properties", []):
            setattr(properties, p, True)

        # non-native is assumed to want symint bindings if you wrote symint
        schema = LazyIrSchema(FunctionSchema.parse(op["func"]), properties, symint=True)
        schema.opkind = op.get("opkind")
        nodes.append(gen_lazy_ir.gen(schema)[0])

    return nodes
