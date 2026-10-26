# [MODULE DOC] Bridges Inductor epilogue source code to CUTLASS Epilogue Visitor Trees (EVT). Defines
# type aliases and, only when the CUTLASS python library can be imported, helpers to build example CUTLASS
# tensors from Inductor buffers, trace epilogue source into an EVT, emit the collective epilogue C++ code,
# and render the C++ argument initializer for the EVT. Import-time side effect: tries to import cutlass.
# When cutlass is unavailable the helper definitions are skipped entirely.
from collections.abc import Callable
from typing import Any

from sympy import Expr

from torch._inductor.ir import (
    ComputedBuffer,
    InputBuffer,
    is_contiguous_strides_for_shape,
)
from torch.utils._ordered_set import OrderedSet

from ..utils import cutlass_arch, torch_dtype_to_cutlass_type, try_import_cutlass


EpilogueFunctor = Any  # EpilogueFunctor local class defined in _trace
Buffer = ComputedBuffer | InputBuffer
CutlassTupleType = Any  # cutlass.backend.c_types.tuple_factory_.<locals>.TupleType
CutlassVisitorType = Any  # cutlass.backend.c_types.visitor_factory.<locals>.VisitorType
CutlassArgType = (
    Any  # Can be a CutlassTupleType, CutlassVisitorType, EmptyByte, or ctype.c_void_p
)


if try_import_cutlass():
    import ast
    import ctypes
    import textwrap

    from cutlass_cppgen.backend.c_types import (  # type: ignore[import-not-found]
        EmptyByte,
    )
    from cutlass_cppgen.backend.epilogue import (  # type: ignore[import-not-found]
        dtype2ctype,
    )
    from cutlass_cppgen.backend.evt import (  # type: ignore[import-not-found]
        backend as evt_backend,
        EpilogueFunctorVisitor,
    )
    from cutlass_cppgen.backend.evt.backend.emitter_base import (  # type: ignore[import-not-found]
        FusionCallbacks,
    )
    from cutlass_cppgen.backend.evt.backend.sm100_emitter import (  # type: ignore[import-not-found]
        Sm100CollectiveEpilogue,
    )
    from cutlass_cppgen.backend.evt.frontend import (  # type: ignore[import-not-found]
        PythonASTFrontend,
    )
    from cutlass_cppgen.backend.evt.ir.tensor import (  # type: ignore[import-not-found]
        Tensor as CutlassTensor,
    )
    from cutlass_library import (
        DataType,
        EpilogueScheduleType,
        LayoutType,
        TileDescription,
    )

    from torch._inductor.utils import IndentedBuffer

    _CUTLASS_C_DTYPES = OrderedSet(dtype2ctype.values())  # type: ignore[var-annotated]

    class EVTArgRenames:
        # [CLASS DOC] Maps Inductor buffer names to generated C++ pointer variable names.
        # State: buf_renames, an insertion-ordered mapping from buffer name to a name of the form ptr_N,
        # where N is the allocation order. Names are stable once assigned.
        """Handles mapping buffer names to variable names in the cpp kernel signature and body"""

        def __init__(self) -> None:
            # [FUNCTION DOC] Creates an empty rename mapping.
            self.buf_renames: dict[str, str] = {}

        def new_name(self, name: str) -> str:
            # [FUNCTION DOC] Returns the existing rename for the buffer name, or allocates and records the
            # next ptr_N name (N is the current mapping size). Mutates the mapping on first sight.
            if name in self.buf_renames:
                return self.buf_renames[name]
            else:
                new_name = f"ptr_{len(self.buf_renames)}"
                self.buf_renames[name] = new_name
                return new_name

        def get(self, name: str) -> str:
            # [FUNCTION DOC] Returns the renamed variable for the buffer name, or the name itself if it
            # has not been renamed. Does not mutate.
            return self.buf_renames.get(name, name)

    def create_example_tensors(
        var_name_to_buffer_name: dict[str, str],
        name_to_buffer: dict[str, Buffer],
        size_hint_fn: Callable[[Expr | int], int],
    ) -> dict[str, CutlassTensor]:
        # [FUNCTION DOC] Builds example CUTLASS tensors for tracing an epilogue. Takes a mapping from
        # epilogue variable name to buffer name, a mapping from buffer name to Inductor buffer, and a
        # function resolving symbolic sizes to integer hints. Returns a mapping from variable name to
        # CUTLASS tensor with the hinted shape, row or column major layout and the buffer's element type.
        # Raises RuntimeError if a buffer is neither row-major nor column-major contiguous.
        def cutlass_tensor_from_buffer(
            buffer: Buffer,
        ) -> CutlassTensor:
            # [FUNCTION DOC] HELPER: Converts one buffer into a CUTLASS tensor using size hints for shape
            # and strides; picks row major or column major from contiguity, else raises RuntimeError.
            shape = buffer.get_layout().size
            stride = buffer.get_layout().stride
            shape = tuple(size_hint_fn(x) for x in shape)
            stride = tuple(size_hint_fn(x) for x in stride)

            is_row_major = is_contiguous_strides_for_shape(stride, shape)
            is_column_major = is_contiguous_strides_for_shape(stride[::-1], shape[::-1])

            if not is_row_major and not is_column_major:
                raise RuntimeError(
                    f"Cannot create example tensor for {buffer.get_name()} with \
non-contiguous layout, received stride: {stride} and shape: {shape}"
                )

            return CutlassTensor(
                shape=shape,
                layout_tag=(
                    LayoutType.RowMajor if is_row_major else LayoutType.ColumnMajor
                ),
                element=torch_dtype_to_cutlass_type(buffer.get_layout().dtype),
            )

        return {
            key: cutlass_tensor_from_buffer(name_to_buffer[name])
            for key, name in var_name_to_buffer_name.items()
        }

    def trace(
        fn_src: str,
        example_tensors: dict[str, CutlassTensor],
        accum_type: DataType,
        output_type: DataType,
        tile_description: TileDescription,
        epilogue_schedule: EpilogueScheduleType,
        name_to_buffer: dict[str, Buffer],
        size_hint_fn: Callable[[Expr | int], int],
        kernel_schedule: Any | None = None,
        device_type: str = "cuda",
        **kwargs: dict[str, Any],
    ) -> tuple[str, str, str, EVTArgRenames]:
        # [FUNCTION DOC] Traces the epilogue source into a CUTLASS EVT and emits its C++ pieces.
        # Inputs: epilogue source text, example tensors, accumulator and output data types, tile
        # description, epilogue schedule, buffer lookup, size hint function, optional kernel schedule (used
        # for SM100+), and device type (cuda or xpu). Returns (EVT type name, C++ argument initializer
        # string, EVT C++ code, argument rename helper). Raises AssertionError for CUDA arch below 90 and
        # NotImplementedError when no emitter exists for the arch (pre-SM100 and xpu use a per-arch
        # emitter; SM100+ uses the SM100 collective epilogue).
        arch = int(cutlass_arch(device_type))
        if device_type == "cuda" and arch < 90:
            raise AssertionError("For CUDA, only SM90+ is supported for EVT")
        epilogue_functor = _trace(fn_src, example_tensors, arch, **kwargs)
        visitor = EpilogueFunctorVisitor(arch, epilogue_functor)
        fusion_callbacks = FusionCallbacks(visitor.graph, arch, emit_CD=False)
        arch_prefix = "xe" if device_type == "xpu" else "sm"

        if device_type == "xpu" or arch < 100:
            try:
                evt_emitter = getattr(evt_backend, f"{arch_prefix}{arch}_emitter")
                CollectiveEpilogue = evt_emitter.CollectiveEpilogue
            except AttributeError as e:
                raise NotImplementedError(
                    f"EVT backend is not supported on Arch {arch_prefix}{arch}."
                ) from e

            collective_epilogue = CollectiveEpilogue(
                tile_description,
                epilogue_schedule,
                accum_type,
                output_type,
                fusion_callbacks,
            )
        else:
            collective_epilogue = Sm100CollectiveEpilogue(
                tile_description=tile_description,
                kernel_schedule=kernel_schedule,
                epilogue_schedule=epilogue_schedule,
                element_accumulator=accum_type,
                element_d=output_type,
                fusion_callbacks=fusion_callbacks,
            )
        evt_name, evt_code = collective_epilogue.emit()
        evt_args, arg_renames = _render_argument_type(
            epilogue_functor, name_to_buffer, size_hint_fn
        )
        return evt_name, evt_args, evt_code, arg_renames

    # Based off of
    # https://github.com/NVIDIA/cutlass/blob/df18f5e4f5de76bed8be1de8e4c245f2f5ec3020/python/cutlass/epilogue/epilogue.py#L117
    # This is modified to enable directly passing the source code of the epilogue vs getting it from a bona-fide python function
    # The reason for this is that inspect.getsource does not work with functions defined at runtime via exec/eval
    def _trace(
        fn_src: str,
        example_tensors: dict[str, CutlassTensor],
        cc: int,
        **kwargs: Any,
    ) -> EpilogueFunctor:
        # [FUNCTION DOC] HELPER: Runs the CUTLASS python AST frontend over epilogue source text (rather than
        # a real function, since source inspection fails for runtime-defined functions). Takes source,
        # example tensors, compute capability and extra frontend kwargs; returns the traced functor.
        class EpilogueFunctor(PythonASTFrontend):
            # [CLASS DOC] HELPER: PythonASTFrontend subclass that parses a stored dedented source string
            # instead of inspecting a function. State: source (the dedented text) and, after parse,
            # example_inputs and ast.
            def __init__(self, cc: int, **kwargs: Any):
                # [FUNCTION DOC] HELPER: Stores dedented epilogue source then initializes the base frontend.
                self.source = textwrap.dedent(fn_src)
                super().__init__(cc, **kwargs)

            def parse(
                self,
                example_inputs: dict[str, CutlassTensor],
            ) -> None:
                # [FUNCTION DOC] HELPER: Records the example inputs, parses the source into a syntax tree
                # and visits it so the base frontend builds its graph. Mutates self.
                self.example_inputs = example_inputs
                self.ast = ast.parse(self.source)
                # pyrefly: ignore [missing-attribute]
                self.visit(self.ast)

        epilogue_functor = EpilogueFunctor(cc=cc, **kwargs)
        epilogue_functor.trace(example_tensors)
        return epilogue_functor

    def _render_argument_type(
        epilogue_functor: EpilogueFunctor,
        name_to_buffer: dict[str, Buffer],
        size_hint_fn: Callable[[Expr | int], int],
    ) -> tuple[str, EVTArgRenames]:
        # [FUNCTION DOC] HELPER: Renders the nested brace-initializer C++ text for the EVT thread
        # arguments by walking the functor's thread type fields recursively. Takes the traced functor,
        # buffer lookup and size hint function. Returns (argument text, rename helper that records
        # pointer names used). Raises RuntimeError for unrecoverable immediate constants.
        epilogue_thread_type = epilogue_functor.epilogue_thread_type
        arg_renames = EVTArgRenames()

        # Fragile, but this is the only way to guarantee t is expected type because t is a local class
        def is_nested_visitor_type(t: type) -> bool:
            # [FUNCTION DOC] HELPER: True if the type is CUTLASS's locally-defined visitor type, detected
            # by comparing its module and qualified name string.
            return (
                ".".join([t.__module__, t.__qualname__])
                == "cutlass_cppgen.backend.c_types.visitor_factory.<locals>.VisitorType"
            )

        buffer = IndentedBuffer()
        with buffer.set_tabwidth(2):

            def get_constant_value(name: str) -> float | int | None:
                # [FUNCTION DOC] HELPER: Finds the constant value for a node name. First consults the
                # dag metadata for a tensor marked constant; otherwise parses names of the form imm_value_kidx
                # (first underscore in the value stands for the decimal point). Returns None if not a
                # recoverable constant.
                dag_ir = getattr(epilogue_functor, "dag_ir", None)
                if dag_ir is not None:
                    try:
                        node_meta = dag_ir.get_node_meta(name)
                    except Exception:
                        node_meta = None
                    if node_meta is not None:
                        tensor_meta = getattr(node_meta, "tensor", None)
                        if isinstance(tensor_meta, dict) and tensor_meta.get(
                            "is_constant"
                        ):
                            return tensor_meta.get("tensor")

                # Immediate constants are encoded as "imm_<value>_k<idx>" where
                # the first underscore in <value> stands in for the decimal point
                # (e.g. "imm_0_044715_k1" -> 0.044715). dag_ir meta is the primary
                # source; this name parsing is a best-effort fallback.
                if not name.startswith("imm_"):
                    return None

                value_part, _, _ = name[len("imm_") :].rpartition("_k")
                if not value_part:
                    return None
                try:
                    return float(value_part.replace("_", ".", 1))
                except ValueError:
                    return None

            def render_argument_type(
                name: str,
                t: CutlassArgType,
                constant_value: float | int | None = None,
            ) -> None:
                # [FUNCTION DOC] HELPER: Writes one argument line to the output buffer. Nodes that are
                # neither buffers nor constants become an empty struct, except unresolved imm_ nodes, which
                # raise RuntimeError. Empty-byte types also give an empty struct; other struct types emit
                # each field via the argument-from-node helper.
                if constant_value is None and name not in name_to_buffer:
                    # An imm_* node is an immediate constant whose value we
                    # failed to recover (dag_ir meta missing AND name parsing
                    # failed, e.g. an unsupported encoding). Emitting an empty
                    # struct here would silently drop the constant and produce a
                    # numerically wrong kernel, so fail loudly instead.
                    if name.startswith("imm_"):
                        raise RuntimeError(
                            f"Could not recover the value of immediate constant "
                            f"node {name!r} for the CUTLASS EVT epilogue. "
                            f"Refusing to emit an empty argument that would "
                            f"silently drop the constant."
                        )
                    # Other non-buffer nodes (e.g. autogen_identity_*/compute_*
                    # placeholders) legitimately render as an empty struct,
                    # matching CUTLASS reference EVT kernels.
                    buffer.writeline(f"{{}}, /* {name} */")
                    return
                if issubclass(t, ctypes.c_byte):
                    buffer.writeline(f"{{}}, /* {name} */")
                else:
                    fields = [
                        (
                            fname,
                            _get_arg_from_node(
                                ty,
                                name_to_buffer.get(name),
                                size_hint_fn,
                                arg_renames,
                                constant_value,
                            ),
                        )
                        for fname, ty in t._fields_
                    ]
                    field_strs = [
                        f"/* {fname} */ {str(field)}" for fname, field in fields
                    ]
                    buffer.writeline(f"{{{', '.join(field_strs)}}}, /* {name} */")

            def render_thread_type(name: str, t: CutlassArgType) -> None:
                # [FUNCTION DOC] HELPER: Recursively writes a nested visitor type as a braced block with
                # a name comment, or writes a leaf argument with its constant value if any.
                if is_nested_visitor_type(t):
                    buffer.writeline(f"{{ /* {name} */")
                    with buffer.indent():
                        for name, inner_t in t._fields_:
                            render_thread_type(name, inner_t)
                    buffer.writeline("},")
                else:
                    render_argument_type(name, t, get_constant_value(name))

            # unroll the recursion once to address special case formatting
            # namely, no ending comma and no indentation for the outermost thread type
            buffer.writeline("{ /* thread */")
            with buffer.indent(3):
                if is_nested_visitor_type(epilogue_thread_type):
                    with buffer.indent():
                        for name, inner_t in epilogue_thread_type._fields_:
                            render_thread_type(name, inner_t)
                else:
                    render_argument_type("thread", epilogue_thread_type)
                buffer.writeline("}")

        return buffer.getvalue(), arg_renames

    def _get_arg_from_node(
        arg_ty: type,
        node: Buffer | None,
        size_hint_fn: Callable[[Expr | int], int],
        arg_renames: EVTArgRenames,
        constant_value: float | int | None = None,
    ) -> str:
        # [FUNCTION DOC] HELPER: Renders one C++ initializer string for a ctypes field type. Stride tuple
        # types give up to three strides (padded with zero, 0 and 1 rendered as compile-time constants);
        # pointer types give a typed pointer expression with offset (nullptr for constants) and register the
        # rename; element dtype types give the constant or a zero of the buffer dtype; empty byte gives
        # empty braces. Raises AssertionError on missing node or long strides and NotImplementedError for
        # constants with tuple types or unsupported types.
        from ..template import CUTLASSTemplate

        # Today, arguments are either a pointer to the
        # node's memory, a stride tuple, the datatype
        # Once again, need to check for local class type for stride tuple
        if (
            str(arg_ty)
            == "<class 'cutlass_cppgen.backend.c_types.tuple_factory_.<locals>.TupleType'>"
        ):
            if constant_value is not None:
                raise NotImplementedError(
                    "Immediate constants do not use tuple argument types"
                )
            DEFAULT_STRIDE_LEN = 3
            if node is None:
                raise AssertionError("node must not be None for tuple argument types")
            if len(node.get_layout().stride) > DEFAULT_STRIDE_LEN:
                raise AssertionError(
                    f"expected stride length <= {DEFAULT_STRIDE_LEN}, "
                    f"got {len(node.get_layout().stride)}"
                )
            stride = [size_hint_fn(x) for x in node.get_layout().stride]
            for _ in range(DEFAULT_STRIDE_LEN - len(stride)):
                stride.append(0)

            def render_stride(x: int) -> str:
                # [FUNCTION DOC] HELPER: Renders a stride value; 0 and 1 become compile-time constant tokens.
                # Handle EBO for 0 and 1
                if x == 0:
                    return "_0{}"
                elif x == 1:
                    return "_1{}"
                else:
                    return str(x)

            return f"{{{', '.join([render_stride(x) for x in stride])}}}"

        elif issubclass(arg_ty, ctypes.c_void_p):
            if constant_value is not None:
                return "nullptr"
            if node is None:
                raise AssertionError("node must not be None for pointer argument")
            name = arg_renames.new_name(node.get_name())
            return f"({CUTLASSTemplate._DTYPE_TO_CUTLASS[node.get_layout().dtype]}*) ({name} + {name}_offset)"
        elif (
            arg_ty in _CUTLASS_C_DTYPES
        ):  # Assumption: this is the element dtype, this holds for all cutlass ir nodes currently
            if constant_value is not None:
                return str(constant_value)
            if node is None:
                raise AssertionError("node must not be None for dtype argument")
            return f"{CUTLASSTemplate._DTYPE_TO_CUTLASS[node.get_layout().dtype]}(0)"
        elif issubclass(arg_ty, EmptyByte):
            return "{}"

        raise NotImplementedError(f"Unsupported arg type: {arg_ty}")
