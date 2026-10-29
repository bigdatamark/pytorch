# mypy: allow-untyped-defs
# [MODULE DOC] Lowerings for oneDNN (mkldnn/onednn) fused ops in Inductor, mostly CPU. Defines int8 compensation helpers,
# the grouped GEMM lowering, a 0-d constant normalizer, and register_onednn_fusion_ops, which (when mkldnn is available)
# registers lowerings for convolution, linear, RNN and quantized conv/linear ops, choosing between C++ GEMM templates
# and ATen mkldnn extern kernels via autotuning. Import-time side effects: tags grouped_gemm_lowering with lowering attributes.
import functools

import torch
import torch.utils._pytree as pytree
from torch._inductor.kernel.mm_common import mm_args

from . import config, ir
from .codegen.cpp_gemm_template import CppGemmTemplate
from .codegen.cpp_grouped_gemm_template import CppGroupedGemmTemplate
from .codegen.cpp_utils import create_epilogue_with_attr
from .ir import TensorBox
from .lowering import (
    add,
    add_needs_realized_inputs,
    aten,
    permute,
    register_lowering,
    to_dtype,
    view,
)
from .select_algorithm import (
    autotune_select_algorithm,
    ChoiceCaller,
    ExternKernelChoice,
)
from .utils import use_aten_gemm_kernels, use_cpp_gemm_template
from .virtualized import ops, OpsValue, V


def create_int8_compensation(
    W_tensor: torch.Tensor,
    packed_weight: ir.TensorBox,
    x_scale: ir.TensorBox,
    x_zp: ir.TensorBox,
    w_scale: ir.TensorBox,
) -> tuple[
    bool,
    ir.TensorBox,
    ir.TensorBox | None,
]:
    # [FUNCTION DOC] Compute the int8 weight compensation constants for a quantized GEMM. If x_scale, x_zp and w_scale are all graph constants
    # (fast path), registers constants for the product of scales and for weight column sums scaled by that product and x_zp;
    # otherwise registers only the plain weight column sums. Returns (fast path flag, compensation tensor, combined
    # scale or None). Side effect: adds tensor constants to the graph, named from the packed weight.
    x_w_scale: ir.TensorBox | None = None
    use_int8_fast_compensation_path = all(
        isinstance(item, ir.TensorBox)
        and item.get_name() in V.graph.constants
        and hasattr(item.data, "data")
        and isinstance(item.data.data, ir.ConstantBuffer)
        for item in [x_scale, x_zp, w_scale]
    )
    if use_int8_fast_compensation_path:
        x_w_scale_tensor = (
            V.graph.constants[x_scale.get_name()]
            * V.graph.constants[w_scale.get_name()]
        )
        x_w_scale = V.graph.add_tensor_constant(
            x_w_scale_tensor,
            name=packed_weight.get_name() + "_x_w_compens",
        )
        weight_compens_tensor = torch.sum(W_tensor.to(torch.float), dim=0)
        x_zp_tensor = V.graph.constants[x_zp.get_name()]
        weight_compens_tensor = weight_compens_tensor * x_w_scale_tensor * x_zp_tensor
        weight_compens = V.graph.add_tensor_constant(
            weight_compens_tensor,
            name=packed_weight.get_name() + "_BMatrixCompens",
        )
    else:
        weight_compens_tensor = torch.sum(W_tensor.to(torch.float), dim=0)
        weight_compens = V.graph.add_tensor_constant(
            weight_compens_tensor,
            name=packed_weight.get_name() + "_BMatrixCompens",
        )
    return (  # type: ignore[return-type]
        use_int8_fast_compensation_path,
        weight_compens,
        x_w_scale,
    )


def codegen_int8_gemm_template_compensation(
    use_int8_fast_compensation_path: bool,
    input: OpsValue,
    _weight_compo: OpsValue,
    _x_scale: OpsValue | None,
    _x_zp: OpsValue | None,
    _w_scale: OpsValue | None,
    _x_w_scale: OpsValue | None,
) -> OpsValue:
    # [FUNCTION DOC] Emit pointwise ops applying int8 compensation to a GEMM output value. Fast path: input times combined scale minus
    # weight compensation. Otherwise: input times x_scale times w_scale minus x_scale times w_scale times x_zp times
    # compensation (applied even when x_zp is zero, to cover dynamic quantization). Returns the resulting ops value.
    if use_int8_fast_compensation_path:
        temp = ops.sub(
            ops.mul(
                input,
                _x_w_scale,
            ),
            _weight_compo,
        )
    else:
        temp = ops.mul(
            ops.mul(
                input,
                _x_scale,
            ),
            _w_scale,
        )
        # NOTE: We will apply compensation even if the x_zp is 0 for int8 quantization.
        # That's because when torch.compile is invoked for dynamic quantization,
        # x might coincidentally have such values that x_zp might be zero despite
        # asymmetric quantization.
        # Besides, if x_zp is dummy for int8 x, or if x is statically quantized,
        # we'd still perform that redundant compute to avoid making the code messy
        # because we discovered that redundant computation of compensation did not
        # lead to performance degradation with the input shapes tested.
        temp = ops.sub(
            temp,
            ops.mul(
                ops.mul(
                    ops.mul(
                        _x_scale,
                        _w_scale,
                    ),
                    _x_zp,
                ),
                _weight_compo,
            ),
        )
    return temp


def grouped_gemm_lowering(
    x: TensorBox,
    w: list[TensorBox],
    b: list[TensorBox],
    attr=None,
    scalars=None,
    algorithm=None,
    layout=None,
):
    # [FUNCTION DOC] Lower a grouped linear (several weights sharing one input) via the C++ grouped GEMM template with autotuning.
    # Flattens inputs with more than two dims, realizes biases, builds choices, selects the best, and exposes each GEMM as a
    # MultiOutput view reshaped to the original leading dims. Returns the list of output tensors.
    # Raises an error if max_autotune is not enabled or no choice is produced.
    x_size = x.get_size()
    if len(x_size) > 2:
        # GEMM template needs 2D input, normalize input shape here
        x = view(x, [-1, x_size[-1]])
    num_gemm = len(w)

    if not (config.max_autotune or config.max_autotune_gemm):
        raise AssertionError(
            "grouped_gemm_lowering requires max_autotune or max_autotune_gemm"
        )
    # pyrefly: ignore [bad-assignment]
    b = [bias if bias is None else ir.ExternKernel.realize_input(bias) for bias in b]

    choices: list[ChoiceCaller] = []
    *_, layout, x, _ = mm_args(x, permute(w[0], [1, 0]), layout=layout)

    kwargs = {
        "has_bias": [bias is not None for bias in b],
        "trans_w": True,
        "epilogue_creator": None,
        "act_mapping": dict.fromkeys(range(num_gemm), x),
    }

    input_nodes = [x, *w]
    input_nodes.extend([bias for bias in b if bias is not None])

    CppGroupedGemmTemplate.add_choices(
        choices,
        layout,
        input_nodes,
        **kwargs,  # type: ignore[arg-type]
    )

    if len(choices) == 0:
        raise AssertionError("expected at least one choice for grouped_gemm")
    result, _ = autotune_select_algorithm(
        "grouped_gemm",
        choices,
        input_nodes,
        layout,
    )
    template_buf = result.data.data
    return_bufs = [
        ir.MultiOutput(layout, template_buf, [(list, gemm_idx)])
        for gemm_idx in range(num_gemm)
    ]
    # pyrefly: ignore [bad-argument-type]
    template_buf.layout = ir.MultiOutputLayout(device=input_nodes[0].get_device())
    template_buf.outputs = return_bufs
    return_tensors = [
        ir.TensorBox.create(return_bufs[gemm_idx]) for gemm_idx in range(num_gemm)
    ]
    if len(x_size) > 2:
        for gemm_idx in range(num_gemm):
            return_tensors[gemm_idx] = view(
                return_tensors[gemm_idx],  # type: ignore[arg-type]
                (*x_size[:-1], return_tensors[gemm_idx].get_size()[-1]),
            )
    return return_tensors


grouped_gemm_lowering._inductor_lowering_function = True  # type: ignore[attr-defined]
grouped_gemm_lowering._inductor_lowering_output_metadata_ignores_input_storage = True  # type: ignore[attr-defined]


def _convert_to_0d_constant(
    tensor_box: ir.TensorBox,
    dtype: torch.dtype,
    name_suffix: str = "_0d",
) -> ir.TensorBox:
    # [FUNCTION DOC] HELPER: Reduce an all-ones-shape tensor to a 0-d constant. Reuses graph constants by reshaping to scalar, or uses a known
    # constant value; otherwise realizes the tensor, makes views contiguous and returns a view to shape []. Side effect:
    # may add tensor constants to the graph and realize the tensor.
    """
    Normalize a tensor with all dimensions equal to 1 to a 0D ConstantBuffer.
    """
    if isinstance(tensor_box, ir.TensorBox):
        data = tensor_box.data
        if isinstance(data, ir.StorageBox):
            data = data.data
    else:
        data = tensor_box

    if isinstance(data, ir.ConstantBuffer):
        tensor = V.graph.constants.get(tensor_box.get_name())
        if tensor is not None:
            return V.graph.add_tensor_constant(
                tensor.reshape([]), name=tensor_box.get_name() + name_suffix
            )

    from torch._inductor.lowering import get_constant_value

    const_value = get_constant_value(data)
    if const_value is not None:
        return V.graph.add_tensor_constant(
            torch.tensor(const_value.value, dtype=dtype).reshape([])
        )

    tensor_box.realize()

    from .ir import ExternKernel, GenericView

    if isinstance(tensor_box.data, GenericView):
        tensor_box = ir.TensorBox(ExternKernel.require_contiguous(tensor_box.data))

    result = view(tensor_box, [])
    return result


def register_onednn_fusion_ops():
    # [FUNCTION DOC] Register the oneDNN fusion lowerings. Does nothing unless mkldnn is available. Creates extern kernel choices for linear and
    # qlinear ops, lists ops that need realized inputs, and defines and registers the nested lowerings below.
    # Side effect: mutates the global lowering registry.
    if torch._C._has_mkldnn:
        from . import mkldnn_ir

        aten_mkldnn_linear_unary = ExternKernelChoice(
            torch.ops.mkldnn._linear_pointwise,
            "mkldnn::_linear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.LinearUnary.create,
        )
        aten_mkldnn_linear_binary = ExternKernelChoice(
            torch.ops.mkldnn._linear_pointwise.binary,
            "mkldnn::_linear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.LinearBinary.create,
        )
        aten_mkldnn_qlinear_unary = ExternKernelChoice(
            torch.ops.onednn.qlinear_pointwise,
            "onednn::qlinear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.QLinearPointwisePT2E.create,
        )
        aten_mkldnn_qlinear_binary = ExternKernelChoice(
            torch.ops.onednn.qlinear_pointwise.binary,
            "onednn::qlinear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.QLinearPointwiseBinaryPT2E.create,
        )
        cpu_needs_realized_inputs: list[
            torch._ops.OpOverload | torch._ops.OpOverloadPacket
        ] = [
            torch.ops.mkldnn._convolution_pointwise,
            torch.ops.mkldnn._convolution_pointwise_,
            torch.ops.mkldnn._convolution_transpose_pointwise,
            torch.ops.mkldnn._linear_pointwise,
            aten.mkldnn_rnn_layer.default,
            torch.ops.onednn.qconv_pointwise,
        ]

        @register_lowering(torch.ops.mkldnn._convolution_pointwise)
        def convolution_unary(
            x: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            attr,
            scalars,
            algorithm,
        ):
            # [FUNCTION DOC] Lower mkldnn unary-fused convolution by creating a ConvolutionUnary node wrapped in a TensorBox.
            return TensorBox.create(
                mkldnn_ir.ConvolutionUnary.create(
                    x,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._convolution_pointwise.binary)
        def convolution_binary(
            x: TensorBox,
            other: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            binary_attr,
            binary_alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ):
            # [FUNCTION DOC] Lower mkldnn convolution with a binary post-op (with unary post-op) by creating a ConvolutionBinary node.
            return TensorBox.create(
                mkldnn_ir.ConvolutionBinary.create(
                    x,
                    other,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    binary_attr,
                    binary_alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._convolution_pointwise_.binary)
        def convolution_binary_inplace(
            x: TensorBox,
            other: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            binary_attr,
            binary_alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ):
            # [FUNCTION DOC] Lower the in-place variant of the binary-fused convolution via ConvolutionBinaryInplace.
            return TensorBox.create(
                mkldnn_ir.ConvolutionBinaryInplace.create(
                    x,
                    other,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    binary_attr,
                    binary_alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._linear_pointwise)
        def linear_unary(
            x: TensorBox,
            w: TensorBox,
            b: TensorBox,
            attr,
            scalars,
            algorithm,
            layout=None,
        ):
            # [FUNCTION DOC] Lower mkldnn linear with unary post-op. Flattens input to 2D, realizes the bias, and if max_autotune is on and the C++ GEMM
            # template applies, adds template choices with an epilogue for the attr. Falls back to (or adds) the ATen mkldnn choice when
            # there are no choices or ATen kernels are preferred. Autotunes (weight is generated from graph constants) and restores the
            # original leading dims. Raises an error if the weight is not a graph constant.
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            if b is not None:
                b = ir.ExternKernel.realize_input(b)  # type: ignore[assignment]
            choices: list[ChoiceCaller] = []
            if config.max_autotune or config.max_autotune_gemm:
                transposed_w = permute(w, [1, 0])
                *_, layout, x, transposed_w = mm_args(x, transposed_w, layout=layout)
                if use_cpp_gemm_template(layout, x, transposed_w):

                    def epilogue_creator(buf):
                    # [FUNCTION DOC] HELPER: Build the post-op epilogue for the GEMM template from attr, scalars and algorithm.
                        return create_epilogue_with_attr(
                            buf, attr, scalars=scalars, algorithm=algorithm
                        )

                    kwargs = {
                        "has_bias": b is not None,
                        "trans_w": True,
                        "epilogue_creator": (
                            None if attr == "none" else epilogue_creator
                        ),
                    }
                    if b is not None:
                        kwargs["input_indices"] = [2, 0, 1]  # type: ignore[assignment]
                    CppGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, w] if b is None else [x, w, b],
                        **kwargs,  # type: ignore[arg-type]
                    )
            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(attr=attr, scalars=scalars, algorithm=algorithm)
                if b is None:
                    kwargs["B"] = None
                choices.append(
                    aten_mkldnn_linear_unary.bind(
                        [x, w] if b is None else [x, w, b],
                        layout,
                        **kwargs,
                    )
                )
            if w.get_name() not in V.graph.constants:
                raise AssertionError("weight must be a graph constant")
            input_gen_fns = {
                1: lambda x: V.graph.constants[x.get_name()],
            }
            result, _ = autotune_select_algorithm(
                "linear_unary",
                choices,
                [x, w] if b is None else [x, w, b],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

        @register_lowering(torch.ops.mkldnn._linear_pointwise.binary)
        def linear_binary(
            x: TensorBox, y: TensorBox, w: TensorBox, b: TensorBox, attr, layout=None
        ):
            # [FUNCTION DOC] Lower mkldnn linear with a binary post-op (other tensor y). Like linear_unary but flattens both x and y, uses a binary epilogue
            # for the C++ template and extern kernel choice, and autotunes. Raises an error if the weight is not a graph constant.
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            y_size = y.get_size()
            if len(y_size) > 2:
                y = view(y, [-1, y_size[-1]])
            if b is not None:
                b = ir.ExternKernel.realize_input(b)  # type: ignore[assignment]
            choices: list[ChoiceCaller] = []
            if config.max_autotune or config.max_autotune_gemm:
                transposed_w = permute(w, [1, 0])
                *_, layout, x, transposed_w, y = mm_args(
                    x, transposed_w, y, layout=layout
                )
                if use_cpp_gemm_template(layout, x, transposed_w):

                    def epilogue_creator(buf):
                    # [FUNCTION DOC] HELPER: Build the binary-add epilogue using y as the other operand.
                        return create_epilogue_with_attr(buf, attr, other=y)

                    kwargs = {
                        "has_bias": b is not None,
                        "trans_w": True,
                        "epilogue_creator": epilogue_creator,
                    }

                    # pyrefly: ignore [bad-typed-dict-key, unsupported-operation]
                    kwargs["input_indices"] = [0, 2, 1] if b is None else [3, 0, 2, 1]
                    CppGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, y, w] if b is None else [x, y, w, b],
                        **kwargs,  # type: ignore[arg-type]
                    )
            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(attr=attr)
                if b is None:
                    kwargs["B"] = None
                choices.append(
                    aten_mkldnn_linear_binary.bind(
                        [x, y, w] if b is None else [x, y, w, b],
                        layout,
                        **kwargs,
                    )
                )
            if w.get_name() not in V.graph.constants:
                raise AssertionError("weight must be a graph constant")
            input_gen_fns = {
                2: lambda x: V.graph.constants[x.get_name()],
            }
            result, _ = autotune_select_algorithm(
                "linear_binary",
                choices,
                [x, y, w] if b is None else [x, y, w, b],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

        @register_lowering(torch.ops.mkldnn._convolution_transpose_pointwise)
        def convolution_transpose_unary(
            x: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            attr,
            scalars,
            algorithm,
        ):
            # [FUNCTION DOC] Lower mkldnn transposed convolution with a unary post-op via ConvolutionTransposeUnary.
            return TensorBox.create(
                mkldnn_ir.ConvolutionTransposeUnary.create(
                    x,
                    weight,
                    bias,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(aten.mkldnn_rnn_layer.default)
        def mkldnn_rnn_layer(
            x: TensorBox,
            w0: TensorBox,
            w1: TensorBox,
            w2: TensorBox,
            w3: TensorBox,
            hx: TensorBox,
            cx: TensorBox,
            reverse: bool,
            batch_sizes: list[int],
            mode: int,
            hidden_size: int,
            num_layers: int,
            has_biases: bool,
            bidirectional: bool,
            batch_first: bool,
            train: bool,
        ):
            # [FUNCTION DOC] Lower mkldnn_rnn_layer via MkldnnRnnLayer, wrapping each output in a TensorBox.
            return pytree.tree_map(
                TensorBox.create,
                mkldnn_ir.MkldnnRnnLayer.create(
                    x,
                    w0,
                    w1,
                    w2,
                    w3,
                    hx,
                    cx,
                    reverse,
                    batch_sizes,
                    mode,
                    hidden_size,
                    num_layers,
                    has_biases,
                    bidirectional,
                    batch_first,
                    train,
                ),
            )

        @register_lowering(torch.ops.onednn.qconv_pointwise, type_promotion_kind=None)
        def qconvolution_unary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp,
            bias: TensorBox,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            attr,
            scalars,
            algorithm,
        ):
            # [FUNCTION DOC] Lower quantized convolution with unary post-op. Converts Python scalar x_scale and x_zp into graph tensor constants,
            # defaults missing zero points to constant zeros, then creates a QConvPointWisePT2E node. Raises an error if x_scale is not
            # a float or x_zp is not an int when not already a tensor.
            if not isinstance(x_scale, ir.TensorBox):
                if type(x_scale) is not float:
                    raise AssertionError(
                        f"expected x_scale to be float, got {type(x_scale)}"
                    )
                x_scale = V.graph.add_tensor_constant(
                    torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
                )

            if x_zp is None:
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="x_zp"
                )
            if not isinstance(x_zp, ir.TensorBox):
                if type(x_zp) is not int:
                    raise AssertionError(f"expected x_zp to be int, got {type(x_zp)}")
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
                )

            if w_zp is None:
                w_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="w_zp"
                )

            return TensorBox.create(
                mkldnn_ir.QConvPointWisePT2E.create(
                    x,
                    x_scale,
                    x_zp,
                    packed_weight,
                    w_scale,
                    w_zp,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(
            torch.ops.onednn.qconv2d_pointwise.binary, type_promotion_kind=None
        )
        @register_lowering(
            torch.ops.onednn.qconv2d_pointwise.binary_tensor, type_promotion_kind=None
        )
        def qconvolution_binary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp,
            accum: TensorBox,
            bias: TensorBox,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            accum_scale,
            accum_zp,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ):
            # [FUNCTION DOC] Lower quantized convolution with binary post-op (registered for binary and binary_tensor overloads). Normalizes x_scale, x_zp
            # and w_zp as in the unary case, and for sum with differing float32/bfloat16 accum and output dtypes converts accum to the
            # output dtype first. Creates QConvPointWiseBinaryPT2E. Raises an error for bad scalar types.
            if not isinstance(x_scale, ir.TensorBox):
                if type(x_scale) is not float:
                    raise AssertionError(
                        f"expected x_scale to be float, got {type(x_scale)}"
                    )
                x_scale = V.graph.add_tensor_constant(
                    torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
                )

            if x_zp is None:
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="x_zp"
                )
            if not isinstance(x_zp, ir.TensorBox):
                if type(x_zp) is not int:
                    raise AssertionError(f"expected x_zp to be int, got {type(x_zp)}")
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
                )

            if w_zp is None:
                w_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="w_zp"
                )

            if (
                binary_attr == "sum"
                and output_dtype in [torch.float32, torch.bfloat16]
                and accum.get_dtype() in [torch.float32, torch.bfloat16]
                and accum.get_dtype() != output_dtype
            ):
                # For int8-mixed-bf16 quantization and inplace add,
                # there is case when accum dtype is float32 but output dtype is bfloat16.
                # Since the accum will be changed in-place with post op sum,
                # we will do accum dtype conversion here.
                accum = to_dtype(accum, output_dtype)
            return TensorBox.create(
                mkldnn_ir.QConvPointWiseBinaryPT2E.create(
                    x,
                    x_scale,  # type: ignore[arg-type]
                    x_zp,  # type: ignore[arg-type]
                    packed_weight,
                    w_scale,
                    w_zp,
                    accum,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    accum_scale,
                    accum_zp,
                    binary_attr,
                    alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithm,
                )
            )

        @register_lowering(torch.ops.onednn.qlinear_pointwise, type_promotion_kind=None)
        def qlinear_unary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp: TensorBox,
            bias: TensorBox,
            o_scale,
            o_zero_point,
            output_dtype,
            attr,
            scalars,
            algorithm,
            layout=None,
        ):
            # [FUNCTION DOC] Lower oneDNN quantized linear with unary post-op. Validates int8/e4m3fn packed weight, flattens x to 2D, normalizes x_scale,
            # x_zp (and w_zp) into tensor constants or 0-d constants, realizes w_scale/w_zp and fixes w_zp dtype to int32. With
            # max_autotune and symmetric weights, adds C++ GEMM template choices whose epilogue applies int8 compensation, bias,
            # unary post-op and requantization to the output dtype; otherwise (or additionally when ATen preferred) adds the ATen
            # qlinear extern kernel choice. Autotunes with graph-constant input generators and restores leading dims.
            # Raises errors for unsupported weight dtype, bad scalar types, non-scalar x_zp, or non-constant packed weight.
            if packed_weight.get_dtype() not in [torch.int8, torch.float8_e4m3fn]:
                raise AssertionError(
                    "Only int8 and e4m3fn weights are supported by oneDNN qlinear."
                )
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            if not isinstance(x_scale, ir.TensorBox):
                if type(x_scale) is not float:
                    raise AssertionError(
                        f"expected x_scale to be float, got {type(x_scale)}"
                    )
                x_scale = V.graph.add_tensor_constant(
                    torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
                )
            else:
                x_scale_size = x_scale.get_size()
                if len(x_scale_size) == 0 or all(dim == 1 for dim in x_scale_size):
                    x_scale = _convert_to_0d_constant(x_scale, torch.float32, "_0d")
                else:
                    x_scale.realize()
                if len(x_scale.get_size()) not in [0, 1]:
                    raise AssertionError("x_scale must be 0D or 1D")

            if x_zp is None:
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="x_zp"
                )
            elif not isinstance(x_zp, ir.TensorBox):
                if type(x_zp) is not int:
                    raise AssertionError(f"expected x_zp to be int, got {type(x_zp)}")
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
                )
            else:
                x_zp_size = x_zp.get_size()
                if len(x_zp_size) == 0 or all(dim == 1 for dim in x_zp_size):
                    # If all outer dims of x_zp are 1, make it a ConstantBuffer with 0D shape
                    # Don't call realize() before _convert_to_0d_constant to preserve ComputedBuffer
                    x_zp = _convert_to_0d_constant(x_zp, torch.int32, "_0d")
                else:
                    x_zp.realize()

            if x_zp.get_numel() != 1:
                raise AssertionError("x_zp is incompatible with oneDNN qlinear")

            # When channels less than 8, w_scale/w_zp is Pointwise instead of ConstantBuffer
            # Refer to
            # https://github.com/pytorch/pytorch/blob/f353d17755ed23b02924c962a86ff99a3405fe10/torch/_inductor/graph.py#L570-L577
            if w_zp is None:
                # If w_zp is None, then it's a dummy tensor created to denote the
                # absence of a zero point, and thus w is int8 symmetrically quantized.
                # Moreover, oneDNN qlinear API doesn't accept None value for zp

                w_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="w_zp"
                )
            w_scale.realize()
            w_zp.realize()
            if w_zp.get_dtype() != torch.int32 and isinstance(
                ir.InputsKernel.unwrap_storage_for_input(w_zp),
                ir.ConstantBuffer,
            ):
                # W_zp might be a ConstantBuffer with int64, convert it to int32
                w_zp_tensor = V.graph.constants[w_zp.get_name()].to(torch.int32)
                w_zp = V.graph.add_tensor_constant(  # type: ignore[assignment]
                    torch.tensor(w_zp_tensor, dtype=torch.int32), name=w_zp.get_name()
                )

            bias_dtype = None if bias is None else bias.get_dtype()
            choices: list[ChoiceCaller] = []

            if config.max_autotune or config.max_autotune_gemm:
                *_, layout, x, packed_weight = mm_args(
                    x, packed_weight, layout=layout, out_dtype=output_dtype
                )

                if (
                    # GEMM template currently only supports symmetrically quantized weights
                    isinstance(
                        ir.InputsKernel.unwrap_storage_for_input(w_zp),
                        ir.ConstantBuffer,
                    )
                    and torch.equal(
                        torch.zeros_like(V.graph.constants[w_zp.get_name()]),
                        V.graph.constants[w_zp.get_name()],
                    )
                ) and use_cpp_gemm_template(layout, x, packed_weight):
                    W_tensor = V.graph.constants[packed_weight.get_name()].to_dense()

                    (
                        use_int8_fast_compensation_path,
                        weight_compens,
                        x_w_scale,
                    ) = create_int8_compensation(
                        W_tensor,
                        packed_weight,
                        x_scale,
                        x_zp,
                        w_scale,
                    )

                    def epilogue_creator(input_buffer):
                        # [FUNCTION DOC] HELPER: Build the C++ GEMM epilogue: convert int32 GEMM output to float32, apply compensation and optional bias via a
                        # Pointwise node, then the unary post-op, then cast or requantize to output_dtype. Returns the final buffer.
                        # Raises an error for unsupported output dtype.
                        # Epilogue to convert from s32 to f32 for u8s8f32
                        if output_dtype not in [
                            torch.float32,
                            torch.bfloat16,
                            torch.uint8,
                            torch.int8,
                        ]:
                            raise AssertionError(
                                f"unexpected output_dtype {output_dtype}"
                            )
                        input_loader = input_buffer.make_loader()
                        weight_compens_loader = weight_compens.make_loader()
                        x_w_scale_loader = None
                        if use_int8_fast_compensation_path:
                            if x_w_scale is None:
                                raise AssertionError(
                                    "x_w_scale must be set on the fast compensation path"
                                )
                            x_w_scale_loader = x_w_scale.make_loader()
                        x_scale_loader = x_scale.make_loader()
                        w_scale_loader = w_scale.make_loader()
                        x_zp_loader = x_zp.make_loader()
                        nonlocal bias
                        bias_loader = None
                        if bias is not None:
                            bias_loader = bias.make_loader()

                        def inner_fn(index):
                            # [FUNCTION DOC] HELPER: Per-element function of the epilogue Pointwise: loads the GEMM value, converts to float32, applies int8
                            # compensation, and adds bias (converted to float32 if bfloat16). Returns the resulting ops value.
                            nonlocal bias
                            input = input_loader(index)
                            # MicroKernel Output is with int32
                            # cvt to FP32 before doing compensation
                            input = ops.to_dtype(input, torch.float32)
                            weight_compens_index = (index[-1],)

                            _x_scale = None
                            _x_zp = None
                            _w_scale = None
                            if not use_int8_fast_compensation_path:
                                _x_scale = x_scale_loader(())
                                _x_zp = x_zp_loader(())
                                _w_scale = w_scale_loader(weight_compens_index)
                            _weight_compo = weight_compens_loader(weight_compens_index)
                            _x_w_scale = None
                            if use_int8_fast_compensation_path:
                                if x_w_scale_loader is None:
                                    raise AssertionError(
                                        "x_w_scale_loader must be set on the fast compensation path"
                                    )
                                _x_w_scale = x_w_scale_loader(weight_compens_index)
                            # Step 1: Compute s8s8->s32 or u8s8->s32 GEMM & then apply compensation
                            temp = codegen_int8_gemm_template_compensation(
                                use_int8_fast_compensation_path,
                                input,
                                _weight_compo,
                                _x_scale,
                                _x_zp,
                                _w_scale,
                                _x_w_scale,
                            )
                            # Step 2: add Bias if applicable
                            if bias is not None:
                                # pyrefly: ignore [not-callable]
                                _bias = bias_loader(weight_compens_index)
                                nonlocal bias_dtype
                                if bias_dtype not in [torch.float32, torch.bfloat16]:
                                    raise AssertionError(
                                        f"unexpected bias_dtype {bias_dtype}"
                                    )
                                if bias_dtype == torch.bfloat16:
                                    _bias = ops.to_dtype(_bias, torch.float32)
                                temp = ops.add(temp, _bias)

                            return temp

                        output_buf = ir.Pointwise(
                            device=input_buffer.get_device(),
                            dtype=torch.float32,  # Hardcode to FP32 for u8s8f32 & s8s8f32
                            inner_fn=inner_fn,
                            ranges=input_buffer.get_size(),
                        )

                        # Step 3: Doing the unary post op fusion
                        if attr != "none":
                            output_buf = create_epilogue_with_attr(
                                output_buf, attr, scalars=scalars, algorithm=algorithm
                            )

                        # Step 4: Cast output to Target Dtype
                        if output_dtype == torch.bfloat16:
                            output_cast_loader = output_buf.make_loader()

                            def inner_fn_cast_output_to_bf16(index):
                                # [FUNCTION DOC] HELPER: Per-element function: load value and convert it to the output dtype (bfloat16).
                                input = output_cast_loader(index)
                                return ops.to_dtype(input, output_dtype)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device_or_error(),
                                dtype=output_dtype,
                                inner_fn=inner_fn_cast_output_to_bf16,
                                ranges=output_buf.get_size(),
                            )
                        elif output_dtype in [torch.uint8, torch.int8]:
                            from .lowering import _create_constants

                            requant_input_loader = output_buf.make_loader()

                            def inner_fn_requant(index, scale, zero_point):
                                # [FUNCTION DOC] HELPER: Per-element function: requantize a float value using scale and zero_point, round, clamp to the uint8 or int8 range per output_dtype
                                # and convert to output_dtype.
                                input = requant_input_loader(index)
                                inv_scale, zero_point = _create_constants(
                                    1.0 / scale, zero_point, dtype=torch.float32
                                )
                                val = ops.round(input * inv_scale) + zero_point
                                if output_dtype == torch.uint8:
                                    qmin, qmax = _create_constants(
                                        0, 255, dtype=torch.float32
                                    )
                                else:
                                    qmin, qmax = _create_constants(
                                        -128, 127, dtype=torch.float32
                                    )
                                clamped = ops.minimum(ops.maximum(val, qmin), qmax)
                                return ops.to_dtype(clamped, output_dtype)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device_or_error(),
                                dtype=output_dtype,
                                inner_fn=functools.partial(
                                    inner_fn_requant,
                                    scale=float(o_scale),
                                    zero_point=int(o_zero_point),
                                ),
                                ranges=output_buf.get_size(),
                            )

                        return output_buf

                    if x.get_dtype() not in [torch.uint8, torch.int8]:
                        raise AssertionError(f"unexpected x dtype {x.get_dtype()}")
                    CppGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, x_scale, x_zp, packed_weight, w_scale, w_zp]
                        if bias is None
                        else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias],
                        has_bias=bias is not None,
                        epilogue_creator=epilogue_creator,
                        input_indices=[0, 3, 1, 2, 4, 5]
                        if bias is None
                        else [6, 0, 3, 1, 2, 4, 5],
                    )
            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(
                    output_scale=o_scale,
                    output_zero_point=o_zero_point,
                    output_dtype=output_dtype,
                    post_op_name=attr,
                    post_op_args=scalars,
                    post_op_algorithm=algorithm,
                )
                if bias is None:
                    kwargs["bias"] = None
                choices.append(
                    aten_mkldnn_qlinear_unary.bind(
                        (x, x_scale, x_zp, packed_weight, w_scale, w_zp)
                        if bias is None
                        else (x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias),
                        layout,
                        **kwargs,
                    )
                )
            if packed_weight.get_name() not in V.graph.constants:
                raise AssertionError("packed_weight must be a graph constant")
            input_gen_fns = {
                3: lambda x: V.graph.constants[x.get_name()],  # packed weight
                4: lambda x: V.graph.constants[x.get_name()],  # weight scale
                5: lambda x: V.graph.constants[x.get_name()],  # weight zp
                6: lambda x: V.graph.constants[x.get_name()],  # bias
            }
            if isinstance(
                ir.InputsKernel.unwrap_storage_for_input(x_scale),
                ir.ConstantBuffer,
            ):
                # x is statically quantized
                input_gen_fns[1] = lambda x: V.graph.constants[x.get_name()]
            if isinstance(
                ir.InputsKernel.unwrap_storage_for_input(x_zp),
                ir.ConstantBuffer,
            ):
                input_gen_fns[2] = lambda x: V.graph.constants[x.get_name()]

            result, _ = autotune_select_algorithm(
                "qlinear_unary",
                choices,
                [x, x_scale, x_zp, packed_weight, w_scale, w_zp]
                if bias is None
                else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

        @register_lowering(
            torch.ops.onednn.qlinear_pointwise.binary, type_promotion_kind=None
        )
        @register_lowering(
            torch.ops.onednn.qlinear_pointwise.binary_tensor, type_promotion_kind=None
        )
        def qlinear_binary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp: TensorBox,
            x2: TensorBox,
            bias: TensorBox,
            o_scale,
            o_zero_point,
            output_dtype,
            x2_scale,
            x2_zp,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
            layout=None,
        ):
            # [FUNCTION DOC] Lower oneDNN quantized linear with binary post-op (add or sum with another tensor x2; registered for binary and
            # binary_tensor overloads). Similar to qlinear_unary: normalizes scales and zero points, converts float x2 to the output dtype for
            # sum when needed, and on the C++ template path (per-tensor activation zero point, symmetric weights) builds an epilogue that
            # compensates, adds bias, adds x2, applies the unary post-op and requantizes; otherwise uses the ATen extern kernel. For sum with a
            # C++ template result whose layout matches x2, re-wraps the result as a view of x2 since x2 is updated in place.
            # Raises errors on rank mismatch, bad scalar types, non-matching sum accum dtype, or non-constant packed weight.
            x_size = x.get_size()
            x2_size = x2.get_size()
            if len(x_size) != len(x2_size):
                raise AssertionError(
                    f"expected x and x2 to have equal rank, got {len(x_size)} and {len(x2_size)}"
                )
            if len(x_size) > 2 and binary_attr in ["add", "sum"]:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
                x2 = view(x2, [-1, x2_size[-1]])
            if not isinstance(x_scale, ir.TensorBox):
                if type(x_scale) is not float:
                    raise AssertionError(
                        f"expected x_scale to be float, got {type(x_scale)}"
                    )
                x_scale = V.graph.add_tensor_constant(
                    torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
                )
            else:
                x_scale_size = x_scale.get_size()
                if len(x_scale_size) == 0 or all(dim == 1 for dim in x_scale_size):
                    x_scale = _convert_to_0d_constant(x_scale, torch.float32, "_0d")
                else:
                    x_scale.realize()
                if len(x_scale.get_size()) not in [0, 1]:
                    raise AssertionError("x_scale must be 0D or 1D")

            if x_zp is None:
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="x_zp"
                )
            elif not isinstance(x_zp, ir.TensorBox):
                if type(x_zp) is not int:
                    raise AssertionError(f"expected x_zp to be int, got {type(x_zp)}")
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
                )
            else:
                x_zp_size = x_zp.get_size()
                if len(x_zp_size) == 0 or all(dim == 1 for dim in x_zp_size):
                    x_zp = _convert_to_0d_constant(x_zp, torch.int32, "_0d")
                else:
                    x_zp.realize()

            if w_zp is None:
                w_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="w_zp"
                )

            # When channels less than 8, w_scale/w_zp is Pointwise instead of ConstantBuffer
            # Refer to
            # https://github.com/pytorch/pytorch/blob/f353d17755ed23b02924c962a86ff99a3405fe10/torch/_inductor/graph.py#L570-L577
            w_scale.realize()
            w_zp.realize()
            if w_zp.get_dtype() != torch.int32 and isinstance(
                ir.InputsKernel.unwrap_storage_for_input(w_zp),
                ir.ConstantBuffer,
            ):
                w_zp_tensor = V.graph.constants[w_zp.get_name()].to(torch.int32)
                w_zp = V.graph.add_tensor_constant(  # type: ignore[assignment]
                    torch.tensor(w_zp_tensor, dtype=torch.int32), name=w_zp.get_name()
                )
            if binary_attr == "sum":
                if output_dtype in [
                    torch.float32,
                    torch.bfloat16,
                ] and x2.get_dtype() in [torch.float32, torch.bfloat16]:
                    if x2.get_dtype() != output_dtype:
                        # For int8-mixed-bf16 quantization and inplace add,
                        # there is case when accum dtype is float32 but output dtype is bfloat16.
                        # Since the accum will be changed in-place with post op sum,
                        # we will do accum dtype conversion here.
                        x2 = to_dtype(x2, output_dtype)
                else:
                    if x2.get_dtype() != output_dtype:
                        raise AssertionError(
                            "dtype of accum for qlinear post op sum should be the same as output"
                        )
            x2_dtype = x2.get_dtype()
            bias_dtype = bias.get_dtype() if bias is not None else None
            choices: list[ChoiceCaller] = []
            if (config.max_autotune or config.max_autotune_gemm) and binary_attr in [
                "add",
                "sum",
            ]:
                *_, layout, x, packed_weight, x2 = mm_args(
                    x, packed_weight, x2, layout=layout, out_dtype=output_dtype
                )
                if (
                    isinstance(
                        ir.InputsKernel.unwrap_storage_for_input(x_zp),
                        ir.ConstantBuffer,
                    )
                    and len(x_zp.get_layout().size) == 0  # Per tensor quant of act
                    and isinstance(
                        ir.InputsKernel.unwrap_storage_for_input(w_zp),
                        ir.ConstantBuffer,
                    )
                    and torch.equal(
                        torch.zeros_like(V.graph.constants[w_zp.get_name()]),
                        V.graph.constants[w_zp.get_name()],
                    )  # We only compensate MatrixB and assume B_zp is 0 to avoid the compensation of MatrixA
                    and use_cpp_gemm_template(layout, x, packed_weight)
                ):
                    W_tensor = V.graph.constants[packed_weight.get_name()]
                    W_tensor = W_tensor.to_dense()
                    (
                        use_int8_fast_compensation_path,
                        weight_compens,
                        x_w_scale,
                    ) = create_int8_compensation(
                        W_tensor,
                        packed_weight,
                        x_scale,
                        x_zp,
                        w_scale,
                    )

                    def epilogue_creator(input_buffer):
                        # [FUNCTION DOC] HELPER: Build the C++ GEMM epilogue for the binary case: compensate and add bias, add x2 (converted to float32), apply the unary post-op,
                        # then cast or requantize to the output dtype. Returns the final buffer. Raises an error for unsupported output dtype.
                        # KNOWN ISSUE: the int8/uint8 requantization path always converts and creates the output as uint8, even when output_dtype is int8.
                        # Epilogue to convert from s32 to f32 for u8s8f32
                        if output_dtype not in [
                            torch.float32,
                            torch.bfloat16,
                            torch.uint8,
                            torch.int8,
                        ]:
                            raise AssertionError(
                                f"unexpected output_dtype {output_dtype}"
                            )

                        input_loader = input_buffer.make_loader()
                        x2_loader = x2.make_loader()
                        weight_compens_loader = weight_compens.make_loader()
                        x_w_scale_loader = None
                        if use_int8_fast_compensation_path:
                            if x_w_scale is None:
                                raise AssertionError(
                                    "x_w_scale must be set on the fast compensation path"
                                )
                            x_w_scale_loader = x_w_scale.make_loader()
                        x_scale_loader = x_scale.make_loader()
                        w_scale_loader = w_scale.make_loader()
                        x_zp_loader = x_zp.make_loader()
                        nonlocal bias
                        bias_loader = None
                        if bias is not None:
                            bias_loader = bias.make_loader()

                        def inner_fn(index):
                            # [FUNCTION DOC] HELPER: Per-element function of the epilogue Pointwise: loads GEMM value and x2, converts to float32, applies int8 compensation,
                            # bias and then adds x2. Returns the resulting ops value.
                            nonlocal bias
                            input = input_loader(index)
                            _x2 = x2_loader(index)
                            _x_scale = None
                            _x_zp = None
                            _w_scale = None
                            weight_compens_index = (index[-1],)
                            if not use_int8_fast_compensation_path:
                                _x_scale = x_scale_loader(())
                                _x_zp = x_zp_loader(())
                                _w_scale = w_scale_loader(weight_compens_index)
                            # MicroKernel Output is with int32: cvt to FP32 before doing compensation
                            input = ops.to_dtype(input, torch.float32)
                            _weight_compo = weight_compens_loader(weight_compens_index)
                            _x_w_scale = None
                            if use_int8_fast_compensation_path:
                                if x_w_scale_loader is None:
                                    raise AssertionError(
                                        "x_w_scale_loader must be set on the fast compensation path"
                                    )
                                _x_w_scale = x_w_scale_loader(weight_compens_index)
                            # Step 1: Doing compensation to cvt fp32
                            temp = codegen_int8_gemm_template_compensation(
                                use_int8_fast_compensation_path,
                                input,
                                _weight_compo,
                                _x_scale,
                                _x_zp,
                                _w_scale,
                                _x_w_scale,
                            )
                            # Step 2: add Bias if applicable
                            if bias is not None:
                                # pyrefly: ignore [not-callable]
                                _bias = bias_loader(weight_compens_index)
                                nonlocal bias_dtype
                                if bias_dtype not in [torch.float32, torch.bfloat16]:
                                    raise AssertionError(
                                        f"unexpected bias_dtype {bias_dtype}"
                                    )
                                if bias_dtype == torch.bfloat16:
                                    _bias = ops.to_dtype(_bias, torch.float32)
                                temp = ops.add(temp, _bias)

                            # Step 3: Binary add
                            nonlocal x2_dtype
                            if x2_dtype not in [torch.float32, torch.bfloat16]:
                                raise AssertionError(f"unexpected x2_dtype {x2_dtype}")
                            if x2_dtype == torch.bfloat16:
                                _x2 = ops.to_dtype(_x2, torch.float32)
                            temp = ops.add(temp, _x2)

                            return temp

                        output_buf = ir.Pointwise(
                            device=input_buffer.get_device(),
                            dtype=torch.float32,  # Hardcode to FP32 for u8s8f32
                            inner_fn=inner_fn,
                            ranges=input_buffer.get_size(),
                        )

                        # Step 4: Unary post op if has
                        if unary_attr != "none":
                            output_buf = create_epilogue_with_attr(
                                output_buf,
                                unary_attr,
                                scalars=unary_scalars,
                                algorithm=unary_algorithm,
                            )

                        # Step 5: Cast output to Target Dtype
                        if output_dtype == torch.bfloat16:
                            output_cast_loader = output_buf.make_loader()

                            def inner_fn_cast_output_to_bf16(index):
                                # [FUNCTION DOC] HELPER: Per-element function: load value and convert it to the output dtype (bfloat16).
                                input = output_cast_loader(index)
                                return ops.to_dtype(input, output_dtype)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device_or_error(),
                                dtype=output_dtype,
                                inner_fn=inner_fn_cast_output_to_bf16,
                                ranges=output_buf.get_size(),
                            )
                        elif output_dtype in [torch.uint8, torch.int8]:
                            from .lowering import _create_constants

                            requant_input_loader = output_buf.make_loader()

                            def inner_fn_requant(index, scale, zero_point):
                                # [FUNCTION DOC] HELPER: Per-element function: requantize a float value with scale and zero_point, round and clamp to the uint8 or int8 range.
                                # KNOWN ISSUE: the final conversion is to uint8 regardless of output_dtype, so int8 outputs would be mishandled.
                                input = requant_input_loader(index)
                                inv_scale, zero_point = _create_constants(
                                    1.0 / scale, zero_point, dtype=torch.float32
                                )
                                val = ops.round(input * inv_scale) + zero_point
                                if output_dtype == torch.uint8:
                                    qmin, qmax = _create_constants(
                                        0, 255, dtype=torch.float32
                                    )
                                else:
                                    qmin, qmax = _create_constants(
                                        -128, 127, dtype=torch.float32
                                    )
                                clamped = ops.minimum(ops.maximum(val, qmin), qmax)
                                return ops.to_dtype(clamped, torch.uint8)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device_or_error(),
                                dtype=torch.uint8,
                                inner_fn=functools.partial(
                                    inner_fn_requant,
                                    scale=float(o_scale),
                                    zero_point=int(o_zero_point),
                                ),
                                ranges=output_buf.get_size(),
                            )

                        return output_buf

                    CppGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2]
                        if bias is None
                        else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2, bias],
                        has_bias=bias is not None,
                        epilogue_creator=epilogue_creator,
                        # Reorder bias and x2
                        input_indices=[0, 3, 1, 2, 4, 5, 6]
                        if bias is None
                        else [7, 0, 3, 1, 2, 4, 5, 6],
                    )

            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(
                    output_scale=o_scale,
                    output_zero_point=o_zero_point,
                    output_dtype=output_dtype,
                    other_scale=x2_scale,
                    other_zp=x2_zp,
                    binary_post_op=binary_attr,
                    binary_alpha=alpha,
                    unary_post_op=unary_attr,
                    unary_post_op_args=unary_scalars,
                    unary_post_op_algorithm=unary_algorithm,
                )
                if bias is None:
                    kwargs["bias"] = None
                choices.append(
                    aten_mkldnn_qlinear_binary.bind(
                        (x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2)
                        if bias is None
                        else (x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2, bias),
                        layout,
                        **kwargs,
                    )
                )
            if packed_weight.get_name() not in V.graph.constants:
                raise AssertionError("packed_weight must be a graph constant")
            input_gen_fns = {
                3: lambda x: V.graph.constants[x.get_name()],
                4: lambda x: V.graph.constants[x.get_name()],
                5: lambda x: V.graph.constants[x.get_name()],
            }
            if bias is not None:
                input_gen_fns[7] = lambda x: V.graph.constants[x.get_name()]  # For bias
            result, _ = autotune_select_algorithm(
                "qlinear_binary",
                choices,
                [x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2]
                if bias is None
                else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2, bias],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if (
                isinstance(result.data.data, ir.CppTemplateBuffer)
                and binary_attr == "sum"
                and result.data.data.layout == x2.get_layout()
            ):
                # In this case, since x2 is inplace updated when binary_attr is "sum"
                # we update the layout of result to view of x2
                result = ir.TensorBox.create(
                    ir.CppTemplateBuffer(
                        layout=ir.NonOwningLayout(
                            ir.ReinterpretView(data=x2, layout=x2.get_layout())
                        ),
                        inputs=result.data.data.inputs,  # type: ignore[arg-type]
                        make_kernel_render=result.data.data.make_kernel_render,  # type: ignore[arg-type]
                        template=result.data.data.template,
                        choice=result.data.data.choice,
                    )
                )
            if len(x_size) > 2 and binary_attr in ["add", "sum"]:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))  # type: ignore[arg-type]
            return result

        if torch._C.has_mkl:
            aten_mkl_linear = ExternKernelChoice(
                torch.ops.mkl._mkl_linear,
                "mkl::_mkl_linear",
                has_out_variant=False,
                kernel_creator=mkldnn_ir.MKLPackedLinear.create,
            )
            cpu_needs_realized_inputs.append(torch.ops.mkl._mkl_linear)

            @register_lowering(torch.ops.mkl._mkl_linear)
            def mkl_packed_linear(
                x: TensorBox,
                packed_w: TensorBox,
                orig_w: TensorBox,
                b: TensorBox | None,
                batch_size,
                *,
                layout=None,
            ):
                # [FUNCTION DOC] Lower mkl packed linear (only registered when MKL is available). With max_autotune and a usable C++ GEMM template, adds
                # template choices; adds the ATen mkl extern kernel choice if no choices or ATen preferred. Autotunes using the original
                # weight to generate inputs for the packed mkldnn weight, then adds the bias b if present. Returns the result tensor.
                # Raises an error if packed or original weight is not a graph constant.
                choices: list[ChoiceCaller] = []
                if config.max_autotune or config.max_autotune_gemm:
                    transposed_w = permute(orig_w, [1, 0])
                    *_, layout, x, transposed_w = mm_args(
                        x, transposed_w, layout=layout
                    )
                    if use_cpp_gemm_template(layout, x, transposed_w):
                        CppGemmTemplate.add_choices(
                            choices,
                            layout,
                            [x, packed_w, orig_w],
                            trans_w=True,
                            input_indices=[0, 2],
                        )

                if len(choices) == 0 or use_aten_gemm_kernels():
                    choices.append(
                        aten_mkl_linear.bind(
                            (x, packed_w, orig_w), layout, B=None, batch_size=batch_size
                        )
                    )

                if packed_w.get_name() not in V.graph.constants:
                    raise AssertionError("packed_w must be a graph constant")
                if orig_w.get_name() not in V.graph.constants:
                    raise AssertionError("orig_w must be a graph constant")
                # packed_w is a mkldnn tensor which we can't generate directly
                # so we use the weights from the original tensor in autotune.
                input_gen_fns = {
                    1: lambda x: V.graph.constants[x.get_name()],
                    2: lambda x: V.graph.constants[x.get_name()],
                }
                result: TensorBox  # annotation on separate line since tuple unpacking doesn't support inline annotation
                result, _ = autotune_select_algorithm(
                    "packed_linear",
                    choices,
                    [x, packed_w, orig_w],
                    layout,
                    input_gen_fns=input_gen_fns,
                )
                if b is not None:
                    result = add(result, b)
                return result

        add_needs_realized_inputs(cpu_needs_realized_inputs)


register_onednn_fusion_ops()
