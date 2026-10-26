# mypy: allow-untyped-defs
# [MODULE DOC] CuTeDSL template plumbing for the FlexGEMM QuACK epilogue. Defines config
# dataclasses (local-reduce, output indices, overall epilogue config), a kernel class
# that renders a generated Python module (main entry plus a compile-only precompile hook
# calling the runtime gemm_epilogue), a caller class that precompiles via the async
# compile pool, and the module-level flex_gemm_epilogue_template instance used by
# lowering. Import-time side effect: creates the template instance.
import dataclasses
import logging
from typing_extensions import override

import torch
from torch._dynamo.utils import counters
from torch._higher_order_ops.flex_gemm import FlexGemmOpSpec
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.cutedsl.cutedsl_kernel import CuteDSLTemplateKernel
from torch._inductor.codegen.cutedsl.cutedsl_template import (
    CuteDSLTemplate,
    CuteDSLTemplateCaller,
)
from torch._inductor.heuristics.template.flex_gemm import GemmConfigKey
from torch._inductor.kernel.flex_gemm.constraints import (
    FlexGemmLocalReduceGeometry,
    FlexGemmOutputContraction,
    LOCAL_REDUCE_COMBINE_FN_SUFFIX,
    LOCAL_REDUCE_FINALIZE_FN_SUFFIX,
)
from torch._inductor.kernel.flex_gemm.runtime import inductor_quack_cache_dir
from torch._inductor.kernel.gemm_epilogue_analysis import GemmOutputLocalReducePlan
from torch._inductor.select_algorithm import PartialRender
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueLocalReduceConfig:
    # [CLASS DOC] Frozen template-time local-reduce metadata: grouped geometry, the
    # template input index of the compressed output (or None), and whether the reduction
    # feeds the main output. Properties expose group, axis and whether physical
    # callbacks are needed.
    """Template-time local-reduce metadata for output and/or feed-main consumers."""

    geometry: FlexGemmLocalReduceGeometry
    out_index: int | None = None
    feeds_main: bool = False

    @classmethod
    def from_output_plan(
        cls,
        local_reduce: GemmOutputLocalReducePlan | None,
        out_index: int | None,
    ) -> "FlexGemmEpilogueLocalReduceConfig | None":
        # [FUNCTION DOC] Build a config from an analyzed output local-reduce plan and
        # the template input index of its output; returns None when there is no plan.
        # Copies the match geometry and the feeds-main flag.
        """Bind analyzed local-reduction consumers to FlexGEMM's runtime ABI."""
        if local_reduce is None:
            return None
        return FlexGemmEpilogueLocalReduceConfig(
            FlexGemmLocalReduceGeometry(
                local_reduce.match.geometry.group,
                local_reduce.match.geometry.axis,
            ),
            out_index,
            local_reduce.feeds_main,
        )

    @property
    def group(self) -> int:
        # [FUNCTION DOC] Property: group size from the geometry.
        return self.geometry.group

    @property
    def axis(self) -> int:
        # [FUNCTION DOC] Property: reduction axis from the geometry.
        return self.geometry.axis

    @property
    def needs_physical_callbacks(self) -> bool:
        # [FUNCTION DOC] Property: whether the geometry requires physical
        # combine/finalize callbacks.
        return self.geometry.needs_physical_callbacks


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueOutputConfig:
    # [CLASS DOC] Frozen record of template input indices of aux outputs plus the
    # optional local-reduce config and output contraction describing all user-visible
    # outputs.
    """Template input indices and plans for all user-visible outputs."""

    aux_out_indices: tuple[int, ...] = ()
    local_reduce: FlexGemmEpilogueLocalReduceConfig | None = None
    output_contraction: FlexGemmOutputContraction | None = None


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueConfig:
    # [CLASS DOC] Frozen per-choice configuration for rendering one QuACK epilogue
    # template: generated epilogue name and source, GEMM op spec, alpha and beta,
    # selected QuACK config key, template input indices and kinds of captured epilogue
    # tensors, and the outputs config (defaults to empty).
    """Metadata needed to render one Inductor-owned QuACK GEMM epilogue choice.

    Attributes:
        epilogue_name: Name of the generated CuTeDSL epilogue callable.
        epilogue_source: Python source that defines ``epilogue_name``.
        gemm_op: Original aten GEMM op spec used to map inputs into QuACK.
        alpha: Static alpha multiplier for addmm/baddbmm inputs.
        beta: Static beta multiplier for addmm/baddbmm bias inputs.
        quack_config_key: Lossless key for the selected QuACK GEMM config.
        epilogue_arg_indices: Template input indices for read-only epilogue captures.
        epilogue_arg_kinds: Broadcast kind for each captured epilogue tensor.
        outputs: Structural plans for auxiliary, reduced, or transformed outputs.
    """

    epilogue_name: str
    epilogue_source: str
    gemm_op: FlexGemmOpSpec
    alpha: float
    beta: float
    quack_config_key: GemmConfigKey
    epilogue_arg_indices: tuple[int, ...] = ()
    epilogue_arg_kinds: tuple[str, ...] = ()
    outputs: FlexGemmEpilogueOutputConfig = dataclasses.field(
        default_factory=FlexGemmEpilogueOutputConfig
    )


class FlexGemmEpilogueKernel(CuteDSLTemplateKernel):
    # [CLASS DOC] Template kernel that renders the complete generated module text for a
    # FlexGEMM epilogue choice, including a compile-only precompile hook; overrides
    # render of the CuTeDSL template kernel and builds call text through private
    # helpers.
    """Render generated FlexGEMM epilogue modules with a compile-only hook."""

    @override
    def render(self, template, **kwargs):
        # [FUNCTION DOC] Render the generated module. Requires a config keyword
        # (RuntimeError on any other option). Registers every template input as a
        # numbered kernel argument, registers the output, assembles the parameter list
        # (inputs, any extra args, stream, device capacity override, quack cache dir),
        # builds the call to the runtime gemm_epilogue with GEMM operands, epilogue and
        # output-plan kwargs, out buffer, expected rank, and the config key flagged as
        # lowering-validated. Emits imports, the epilogue source, a main function and a
        # precompile function that creates empty strided tensors from metadata and calls
        # main in compile-only mode. Side effect: sets internal input-arg state on the
        # kernel. Returns a PartialRender of the code with the kernel's render hooks.
        config = kwargs.pop("config")
        if kwargs:
            raise RuntimeError(f"unexpected FlexGEMM epilogue options: {kwargs}")

        self._template_input_args = []
        self._seen_input_args = OrderedSet()
        for index, input_node in enumerate(self.input_nodes):
            buf_name = input_node.get_name()
            self.args.input(buf_name)
            arg_name = f"arg{index}"
            self.args.input_buffers[buf_name] = arg_name
            self._template_input_args.append((arg_name, input_node))
            self._seen_input_args.add(arg_name)

        self.args.output(self.output_node.get_name())
        arg_defs, _, _, _ = self.args.python_argdefs()
        params = [arg_name for arg_name, _ in self._template_input_args]
        for arg_def in arg_defs:
            if arg_def.full_name() not in self._seen_input_args:
                params.append(arg_def.full_name())
        params.append("stream")
        params.append("device_capacity_override=None")
        quack_cache_dir_param = f"quack_cache_dir={inductor_quack_cache_dir()!r}"
        params.append(quack_cache_dir_param)

        template_input_arg_names = [
            arg_name for arg_name, _ in self._template_input_args
        ]
        # Template inputs include GEMM operands plus closed-over epilogue tensors for reads and aux writes.
        call_args, call_kwargs = self._gemm_call_args(template_input_arg_names, config)
        call_kwargs += self._epilogue_kwargs(template_input_arg_names, config)
        call_kwargs += (
            f", out={self.get_output()}, "
            f"expected_ndim={config.gemm_op.input_ndim}, "
            "stream=stream, "
            "device_capacity_override=device_capacity_override, "
            "quack_cache_dir=quack_cache_dir"
        )
        call_kwargs += (
            f", config_key={config.quack_config_key!r}, "
            "config_is_lowering_validated=True"
        )

        output_name = self.get_output()

        code = IndentedBuffer()
        code.splice(
            """
            import torch
            from torch._inductor.kernel.flex_gemm.constraints import (
                FlexGemmOutputContraction,
                FlexGemmLocalReduceCallbacks,
                FlexGemmLocalReduceGeometry,
            )
            from torch._inductor.kernel.flex_gemm.runtime import (
                FlexGemmRuntimeLocalReducePlan,
                FlexGemmRuntimeOutputPlan,
                gemm_epilogue as flex_gemm_epilogue,
            )
            """
        )
        code.splice(config.epilogue_source)
        code.splice(
            f"""
            def {self.kernel_name}_main({", ".join(params)}):
                flex_gemm_epilogue(
                    {", ".join(call_args)},
                    {config.epilogue_name},
                    {config.epilogue_name!r}{call_kwargs},
                )

            def {self.kernel_name}_precompile(
                precompile_shapes,
                precompile_strides,
                precompile_dtypes,
                device_index=0,
                device_capability=None,
                hw_info=None,
                quack_cache_dir={inductor_quack_cache_dir()!r},
            ):
                from torch._vendor.quack.cache import compile_only_mode

                device = f"cuda:{{device_index}}"
                with compile_only_mode():
            """
        )
        with code.indent():
            with code.indent():
                for arg_name, _ in self._template_input_args:
                    code.writeline(
                        f"{arg_name} = torch.empty_strided(tuple(precompile_shapes[{arg_name!r}]), "
                        f"tuple(precompile_strides[{arg_name!r}]), device=device, "
                        f"dtype=getattr(torch, precompile_dtypes[{arg_name!r}]))"
                    )
                code.writeline(
                    f"{output_name} = torch.empty_strided(tuple(precompile_shapes['output']), "
                    "tuple(precompile_strides['output']), device=device, "
                    "dtype=getattr(torch, precompile_dtypes['output']))"
                )
                code.writeline(f"{self.kernel_name}_main(")
                with code.indent():
                    for arg_name in template_input_arg_names:
                        code.writeline(f"{arg_name},")
                    code.writeline(f"{output_name}={output_name},")
                    code.writeline("stream=None,")
                    code.writeline("device_capacity_override=device_capability,")
                    code.writeline("quack_cache_dir=quack_cache_dir,")
                code.writeline(")")
        return PartialRender(code.getvalue(), self.render_hooks)

    def _gemm_call_args(
        self, input_args: list[str], config: FlexGemmEpilogueConfig
    ) -> tuple[list[str], str]:
        # [FUNCTION DOC] HELPER: Return the positional GEMM operand names (mat1, mat2)
        # and, if the op has a bias input, a keyword text passing C, alpha and beta;
        # otherwise an empty text.
        """Return positional GEMM operands and scalar/bias kwargs for runtime dispatch."""
        op = config.gemm_op
        call_args = [input_args[op.mat1_index], input_args[op.mat2_index]]
        if op.bias_index is None:
            return call_args, ""
        return call_args, (
            f", C={input_args[op.bias_index]}, alpha={config.alpha!r}, beta={config.beta!r}"
        )

    def _local_reduce_callbacks(self, epilogue_name: str) -> str:
        # [FUNCTION DOC] HELPER: Return source text constructing the callbacks object
        # from the combine and finalize function names derived from the epilogue name.
        """Render generated physical reducer callbacks for runtime registration."""
        combine_name = f"{epilogue_name}{LOCAL_REDUCE_COMBINE_FN_SUFFIX}"
        finalize_name = f"{epilogue_name}{LOCAL_REDUCE_FINALIZE_FN_SUFFIX}"
        return (
            "FlexGemmLocalReduceCallbacks("
            f"combine_fn={combine_name}, finalize_fn={finalize_name})"
        )

    def _local_reduce_geometry(
        self, local_reduce: FlexGemmEpilogueLocalReduceConfig
    ) -> str:
        # [FUNCTION DOC] HELPER: Return source text constructing the local-reduce
        # geometry (group and axis).
        """Render the shared grouped M/N local-reduce geometry."""
        return (
            "FlexGemmLocalReduceGeometry("
            f"group={local_reduce.group!r}, axis={local_reduce.axis!r})"
        )

    def _local_reduce_expr(
        self,
        input_args: list[str],
        local_reduce: FlexGemmEpilogueLocalReduceConfig | None,
        epilogue_name: str,
    ) -> str:
        # [FUNCTION DOC] HELPER: Return source text constructing a runtime local-reduce
        # plan (geometry, optional output argument, feeds-main flag, callbacks when
        # feeding main or when physical callbacks are needed), or the text for none when
        # the config is None.
        """Render one structural local-reduce plan for runtime dispatch."""
        if local_reduce is None:
            return "None"
        geometry = self._local_reduce_geometry(local_reduce)
        plan = f"FlexGemmRuntimeLocalReducePlan({geometry}"
        if local_reduce.out_index is not None:
            plan += f", out={input_args[local_reduce.out_index]}"
        if local_reduce.feeds_main:
            plan += ", feeds_main=True"
        if local_reduce.feeds_main or local_reduce.needs_physical_callbacks:
            plan += f", callbacks={self._local_reduce_callbacks(epilogue_name)}"
        return f"{plan})"

    def _output_plan_expr(
        self, input_args: list[str], config: FlexGemmEpilogueConfig
    ) -> str:
        # [FUNCTION DOC] HELPER: Return source text constructing the runtime output plan
        # from aux output arguments, the local-reduce expression and the output
        # contraction (group and chunked), using none for absent parts.
        """Render all output consumers into one runtime ABI value."""
        outputs = config.outputs
        aux_outs = tuple(input_args[index] for index in outputs.aux_out_indices)
        aux_expr = f"({', '.join(aux_outs)},)" if aux_outs else "()"
        local_reduce_expr = self._local_reduce_expr(
            input_args, outputs.local_reduce, config.epilogue_name
        )
        contraction = outputs.output_contraction
        contraction_expr = (
            "None"
            if contraction is None
            else "FlexGemmOutputContraction("
            f"group={contraction.group!r}, chunked={contraction.chunked!r})"
        )
        return (
            "FlexGemmRuntimeOutputPlan("
            f"aux_outs={aux_expr}, local_reduce={local_reduce_expr}, "
            f"output_contraction={contraction_expr})"
        )

    def _epilogue_kwargs(
        self, input_args: list[str], config: FlexGemmEpilogueConfig
    ) -> str:
        # [FUNCTION DOC] HELPER: Return extra keyword-argument text only for non-default
        # features: captured epilogue args with their kinds, and an output plan when aux
        # outputs, a local reduce or a contraction exist; empty text otherwise.
        """Render only values that differ from the runtime ABI defaults."""
        epilogue_args = [input_args[index] for index in config.epilogue_arg_indices]
        kwargs: list[str] = []
        if epilogue_args:
            kwargs.append(
                f", epilogue_args=({', '.join(epilogue_args)},), "
                f"epilogue_arg_kinds={config.epilogue_arg_kinds!r}"
            )
        outputs = config.outputs
        if (
            outputs.aux_out_indices
            or outputs.local_reduce is not None
            or outputs.output_contraction is not None
        ):
            kwargs.append(f", output_plan={self._output_plan_expr(input_args, config)}")
        return "".join(kwargs)


class FlexGemmEpilogueCaller(CuteDSLTemplateCaller):
    # [CLASS DOC] Template caller specialized for FlexGEMM; adds precompilation of the
    # generated module's QuACK objects using tensor metadata from the benchmark request.
    def precompile(self) -> None:
        # [FUNCTION DOC] Precompile the generated module in the async compile process
        # pool if one is available (waits for the pool to be ready first). Does nothing
        # when metadata is unavailable (dynamic shapes) or no process pool is used.
        # Blocks until the compile finishes; side effect: warms the QuACK cache.
        """Warm the generated FlexGEMM epilogue module's QuACK object cache."""
        metadata = self.precompile_metadata()
        if metadata is None:
            return
        from torch._inductor.async_compile import AsyncCompile

        if not AsyncCompile.use_process_pool():
            AsyncCompile.wait_pool_ready()
        if not AsyncCompile.use_process_pool():
            return
        AsyncCompile().cutedsl(
            self.bmreq.kernel_name,
            self.bmreq.source_code,
            precompile_metadata=metadata,
        ).result()

    def precompile_metadata(self) -> dict[str, object] | None:
        # [FUNCTION DOC] Build the mapping passed to the generated precompile hook: per-
        # tensor shapes, strides and dtype names for inputs (named by position) and the
        # output, device index, device capability (if CUDA is available) and the QuACK
        # cache dir. If any size or stride is symbolic, increments a counter, logs at
        # debug level and returns None.
        """Build the generated FlexGEMM precompile hook's tensor metadata."""
        precompile_shapes = {}
        precompile_strides = {}
        precompile_dtypes = {}
        tensor_metas = [
            *(
                (f"arg{index}", tensor_meta)
                for index, tensor_meta in enumerate(self.bmreq.input_tensor_meta)
            ),
            ("output", self.bmreq.output_tensor_meta),
        ]
        try:
            for name, tensor_meta in tensor_metas:
                precompile_shapes[name] = [int(size) for size in tensor_meta.sizes]
                precompile_strides[name] = [
                    int(stride) for stride in tensor_meta.strides
                ]
                precompile_dtypes[name] = str(tensor_meta.dtype).removeprefix("torch.")
        except (TypeError, RuntimeError, ValueError):
            counters["inductor"]["flex_gemm_precompile_skipped_dynamic"] += 1
            log.debug("Skipping FlexGEMM precompile for symbolic tensor metadata")
            return None
        device_index = self.bmreq.output_tensor_meta.device.index or 0
        device_capability = None
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability(device_index)
        return {
            "precompile_shapes": precompile_shapes,
            "precompile_strides": precompile_strides,
            "precompile_dtypes": precompile_dtypes,
            "device_index": device_index,
            "device_capability": device_capability,
            "quack_cache_dir": inductor_quack_cache_dir(),
        }


class FlexGemmEpilogueTemplate(CuteDSLTemplate):
    # [CLASS DOC] CuTeDSL template for FlexGEMM epilogues; uses the
    # FlexGemmEpilogueKernel and FlexGemmEpilogueCaller classes and has an empty source
    # because the kernel renders the module itself.
    kernel_type = FlexGemmEpilogueKernel
    caller_type = FlexGemmEpilogueCaller

    def __init__(self) -> None:
        # [FUNCTION DOC] Initialize the template under the name flex_gemm_epilogue with
        # empty source.
        super().__init__("flex_gemm_epilogue", source="")


flex_gemm_epilogue_template = FlexGemmEpilogueTemplate()
