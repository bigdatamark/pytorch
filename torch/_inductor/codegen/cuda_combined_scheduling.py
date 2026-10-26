# [MODULE DOC] Defines CUDACombinedScheduling, the scheduling backend registered for CUDA devices that routes
# nodes to Triton, CUTLASS, ROCm C++, CuteDSL or NV universal GEMM schedulers. Creates a module logger at import.
# mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from ..ir import MultiTemplateBuffer
from ..scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)
from .cutedsl.cutedsl_scheduling import CuteDSLScheduling
from .cutlass.scheduling import CUTLASSScheduling
from .nv_universal_gemm.nv_universal_gemm_scheduling import NVUniversalGemmScheduling
from .rocm.rocm_cpp_scheduling import ROCmCPPScheduling
from .triton import TritonScheduling


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from sympy import Expr

    import torch
    from torch.utils._ordered_set import OrderedSet

    from .common import BackendFeature

    _IntLike: TypeAlias = int | Expr


class CUDACombinedScheduling(BaseScheduling):
    # [CLASS DOC] Scheduling facade for CUDA devices. Holds one sub-scheduler each for Triton, CUTLASS, ROCm C++,
    # CuteDSL and NVIDIA Universal GEMM (all built from the same Scheduler), picks one per node (choose_node_backend)
    # and delegates fusion decisions, template codegen and benchmarking. Everything not template-specific goes to Triton.
    """
    Scheduler for CUDA Kernels, which delegates calls as appropriate
    to the CUDA-C++ and Triton Schedulers, which both work for CUDA devices
    and use a unified-wrapper for codegen.

    If Scheduling code needs to be specialized for the case of mixed Triton / CUDA C++ code,
    this would also be the place to do it.
    """

    def __init__(self, scheduler: Scheduler | None) -> None:
        # [FUNCTION DOC] Creates the five backend sub-schedulers from the given scheduler (which may be None).
        super().__init__(scheduler)
        self._triton_scheduling = TritonScheduling(scheduler)
        self._cutlass_scheduling = CUTLASSScheduling(scheduler)
        self._rocm_cpp_scheduling = ROCmCPPScheduling(scheduler)
        self._cutedsl_scheduling = CuteDSLScheduling(scheduler)
        self._nv_universal_gemm_scheduling = NVUniversalGemmScheduling(scheduler)

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        # [FUNCTION DOC] Returns the Triton scheduler's backend features for the device.
        return self._triton_scheduling.get_backend_features(device)

    def choose_node_backend(self, node: BaseSchedulerNode) -> BaseScheduling:
        # [FUNCTION DOC] Returns the sub-scheduler for a node, checked in order: CUTLASS template, ROCm C++ template, CuteDSL template,
        # NV universal GEMM template or fused template; default is Triton.
        if self._cutlass_scheduling.is_cutlass_template(node):
            return self._cutlass_scheduling
        if self._rocm_cpp_scheduling.is_rocm_cpp_template(node):
            return self._rocm_cpp_scheduling
        if self._cutedsl_scheduling.is_cutedsl_template(node):
            return self._cutedsl_scheduling
        if self._nv_universal_gemm_scheduling.is_nv_universal_gemm_template(node):
            return self._nv_universal_gemm_scheduling
        if self._nv_universal_gemm_scheduling.is_nv_universal_gemm_fused_template(node):
            return self._nv_universal_gemm_scheduling
        return self._triton_scheduling

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Decides vertical fusion. CUTLASS may approve; otherwise any CUTLASS or CuteDSL template node blocks fusion.
        # If node1 is an NV universal GEMM template, defer to that backend, falling back to Triton when node1 is a multi-template
        # buffer and NVGEMM refuses. If node1 is an NVGEMM fused template, refuse when node2 is also one, else defer to NVGEMM.
        # All other cases go to Triton.
        if self._cutlass_scheduling.can_fuse_vertical(node1, node2):
            return True
        elif self._cutlass_scheduling.is_cutlass_template(
            node1
        ) or self._cutlass_scheduling.is_cutlass_template(node2):
            return False
        # CuteDSL doesn't support vertical fusion currently
        elif self._cutedsl_scheduling.is_cutedsl_template(
            node1
        ) or self._cutedsl_scheduling.is_cutedsl_template(node2):
            return False
        # Only intercept when node1 is the NVGEMM template (epilogue direction).
        # Prologue direction (node1=pointwise, node2=template) must fall through to
        # Triton, or NVGEMM-winning MTBs silently lose Triton prologue fusion.
        # For MTBs, if NVGEMM can't fuse, fall through to Triton scheduling so
        # Triton choices in the same MTB can still attempt epilogue fusion.
        elif self._nv_universal_gemm_scheduling.is_nv_universal_gemm_template(node1):
            if self._nv_universal_gemm_scheduling.can_fuse_vertical(node1, node2):
                return True
            if isinstance(node1, SchedulerNode) and isinstance(
                node1.node, MultiTemplateBuffer
            ):
                return self._triton_scheduling.can_fuse_vertical(node1, node2)
            return False
        elif self._nv_universal_gemm_scheduling.is_nv_universal_gemm_fused_template(
            node1
        ):
            if self._nv_universal_gemm_scheduling.is_nv_universal_gemm_fused_template(
                node2
            ):
                return False
            return self._nv_universal_gemm_scheduling.can_fuse_vertical(node1, node2)
        return self._triton_scheduling.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Checks node1 then node2: if a node is a CUTLASS, CuteDSL or NVGEMM (plain or fused) template, returns that backend's
        # answer (currently always False); otherwise asks Triton.
        for node in (node1, node2):
            if self._cutlass_scheduling.is_cutlass_template(node):
                return self._cutlass_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
            if self._cutedsl_scheduling.is_cutedsl_template(node):
                return self._cutedsl_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
            if self._nv_universal_gemm_scheduling.is_nv_universal_gemm_template(
                node
            ) or self._nv_universal_gemm_scheduling.is_nv_universal_gemm_fused_template(
                node
            ):
                return self._nv_universal_gemm_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
        return self._triton_scheduling.can_fuse_horizontal(node1, node2)

    def can_fuse_reduction_epilogue(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Delegates to NV universal GEMM when node1 is an NVGEMM template (plain or fused); False for everything else.
        if self._nv_universal_gemm_scheduling.is_nv_universal_gemm_template(
            node1
        ) or self._nv_universal_gemm_scheduling.is_nv_universal_gemm_fused_template(
            node1
        ):
            return self._nv_universal_gemm_scheduling.can_fuse_reduction_epilogue(
                node1, node2
            )
        return False

    def group_fn(
        self, sizes: Sequence[Sequence[_IntLike]]
    ) -> tuple[tuple[_IntLike, ...], ...]:
        # [FUNCTION DOC] Delegates size grouping to the Triton scheduler.
        return self._triton_scheduling.group_fn(sizes)

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> str | None:
        # [FUNCTION DOC] Generates code for a template node plus fused epilogue/prologue nodes using the backend that owns the template
        # (CUTLASS, ROCm C++, CuteDSL, NVGEMM, else Triton) and returns its result (source or None). Raises AssertionError when the
        # backend does not support the given fusion: CUTLASS no prologue; ROCm no epilogue or prologue; CuteDSL no epilogue or
        # prologue; NVGEMM no prologue.
        if self._cutlass_scheduling.is_cutlass_template(template_node):
            if prologue_nodes:
                raise AssertionError("cutlass template does not support prologue nodes")
            return self._cutlass_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        elif self._rocm_cpp_scheduling.is_rocm_cpp_template(template_node):
            if epilogue_nodes:
                raise AssertionError(
                    "rocm cpp template does not support epilogue nodes"
                )
            if prologue_nodes:
                raise AssertionError(
                    "rocm cpp template does not support prologue nodes"
                )
            return self._rocm_cpp_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        elif self._cutedsl_scheduling.is_cutedsl_template(template_node):
            # TODO remove this when we add epilogue support
            if epilogue_nodes:
                raise AssertionError("cutedsl template does not support epilogue nodes")
            if prologue_nodes:
                raise AssertionError("cutedsl template does not support prologue nodes")
            return self._cutedsl_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        elif self._nv_universal_gemm_scheduling.is_nv_universal_gemm_template(
            template_node
        ):
            if prologue_nodes:
                raise AssertionError(
                    "NVIDIA Universal GEMM doesn't support prologue fusion yet"
                )
            return self._nv_universal_gemm_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        else:
            return self._triton_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )

    def codegen_mix_order_reduction(self, node):
        # [FUNCTION DOC] Delegates to the Triton scheduler.
        return self._triton_scheduling.codegen_mix_order_reduction(node)

    def codegen_nested_reduction(self, node):
        # [FUNCTION DOC] Delegates to the Triton scheduler.
        return self._triton_scheduling.codegen_nested_reduction(node)

    def codegen_node(self, node: FusedSchedulerNode | SchedulerNode) -> None:
        # [FUNCTION DOC] Delegates codegen of a (fused) node to the Triton scheduler.
        return self._triton_scheduling.codegen_node(node)

    def codegen_sync(self) -> None:
        # [FUNCTION DOC] Delegates to the Triton scheduler.
        return self._triton_scheduling.codegen_sync()

    def flush(self) -> None:
        # [FUNCTION DOC] Delegates flush to the Triton scheduler.
        return self._triton_scheduling.flush()

    def codegen_combo_kernel(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Delegates combo kernel codegen to the Triton scheduler.
        return self._triton_scheduling.codegen_combo_kernel(*args, **kwargs)

    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[float, str]:
        # [FUNCTION DOC] Delegates benchmarking of fused nodes to the Triton scheduler; returns (time, path).
        return self._triton_scheduling.benchmark_fused_nodes(nodes)

    def benchmark_codegened_module(self, module):
        # [FUNCTION DOC] Benchmarks a generated module: modules flagged as NVGEMM (attribute is_nvgemm) use the local NVGEMM benchmark,
        # others go to Triton. Returns (time in ms, module file path).
        if getattr(module, "is_nvgemm", False):
            return self._benchmark_nvgemm_module(module)
        return self._triton_scheduling.benchmark_codegened_module(module)

    def _benchmark_nvgemm_module(self, module) -> tuple[float, str]:
        # [FUNCTION DOC] HELPER: Benchmarks an NVGEMM fused module on the current graph device while preserving RNG state. Builds args, runs
        # the call once to trigger compilation, then times it. Any exception in either phase is logged at debug level and
        # yields (infinity, module file or empty string). Returns (milliseconds, module file path).
        from torch._dynamo.utils import preserve_rng_state
        from torch._inductor.runtime.benchmarking import benchmarker
        from torch._inductor.utils import get_interface_for_device
        from torch._inductor.virtualized import V

        device_interface = get_interface_for_device(V.graph.device_type)

        with (
            preserve_rng_state(),
            device_interface.device(V.graph.get_current_device_or_throw()),
        ):
            args = module.get_args()
            call = module.call

            try:
                call(args)
            except Exception as e:
                log.debug(
                    "Exception (%s) in compiling NVGEMM fused kernel",
                    e,
                )
                return float("inf"), module.__file__ or ""

            device = V.graph.get_current_device_or_throw()
            try:
                ms = benchmarker.benchmark(
                    lambda: call(args),
                    device=device,
                )
            except Exception as e:
                log.debug(
                    "Exception (%s) while benchmarking NVGEMM fused kernel",
                    e,
                )
                return float("inf"), module.__file__ or ""

            return ms, module.__file__ or ""

    def _is_nvgemm_node(self, node) -> bool:
        # [FUNCTION DOC] HELPER: True if the node's template node is an NV universal GEMM buffer, or a multi-template buffer whose current
        # render kind is nvgemm.
        """Check if a template node is currently configured for NVGEMM codegen."""
        from torch._inductor.ir import MultiTemplateBuffer, NVUniversalGemmBuffer

        template_node = node.get_template_node()
        if isinstance(template_node, NVUniversalGemmBuffer):
            return True
        if isinstance(template_node, MultiTemplateBuffer):
            return template_node._render_kind == "nvgemm"
        return False

    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence[Any],
        benchmark_kernel: bool = False,
        hint_override: int | None = None,
    ) -> str:
        # [FUNCTION DOC] Generates kernel source for the nodes. If the first template node found is configured for NVGEMM codegen, delegates
        # to the NVGEMM scheduler; otherwise to Triton. Only the first template node is inspected. Returns source string.
        for node in nodes:
            if not (hasattr(node, "is_template") and node.is_template()):
                continue
            if self._is_nvgemm_node(node):
                return (
                    self._nv_universal_gemm_scheduling.generate_kernel_code_from_nodes(
                        nodes, benchmark_kernel, hint_override=hint_override
                    )
                )
            break
        return self._triton_scheduling.generate_kernel_code_from_nodes(
            nodes, benchmark_kernel, hint_override=hint_override
        )

    def benchmark_combo_kernel(
        self, node_list: Sequence[BaseSchedulerNode], node_benchmark_results
    ) -> tuple[float, float, list[str | None]]:
        # [FUNCTION DOC] Delegates combo-kernel benchmarking to the Triton scheduler.
        return self._triton_scheduling.benchmark_combo_kernel(
            node_list, node_benchmark_results
        )
