# [MODULE DOC] Defines XPUCombinedScheduling, a scheduling backend for XPU that routes nodes to either the
# CUTLASS (SYCL C++) scheduling or the Triton scheduling. No import-time side effects beyond imports.
# mypy: allow-untyped-defs
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from torch._inductor.scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)

from ..cutlass.scheduling import CUTLASSScheduling
from ..triton import TritonScheduling


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from sympy import Expr

    import torch
    from torch.utils._ordered_set import OrderedSet

    from ..common import BackendFeature

    _IntLike: TypeAlias = int | Expr


class XPUCombinedScheduling(BaseScheduling):
    # [CLASS DOC] Scheduling backend for XPU devices. Holds one TritonScheduling and one CUTLASSScheduling built from the
    # same scheduler. CUTLASS template nodes go to the CUTLASS backend; everything else delegates to Triton.
    """
    Scheduler for XPU Kernels, which delegates calls as appropriate
    to the SYCL-C++ and Triton Schedulers, which both work for XPU devices
    and use a unified-wrapper for codegen.

    If Scheduling code needs to be specialized for the case of mixed Triton / SYCL C++ code,
    this would also be the place to do it.
    """

    def __init__(self, scheduler: Scheduler | None) -> None:
        # [FUNCTION DOC] Initialize the base scheduling with the scheduler and create the Triton and CUTLASS sub-schedulings.
        super().__init__(scheduler)
        self._triton_scheduling = TritonScheduling(scheduler)
        self._cutlass_scheduling = CUTLASSScheduling(scheduler)

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        # [FUNCTION DOC] Returns the backend feature set of the Triton scheduling for the given device.
        return self._triton_scheduling.get_backend_features(device)

    def choose_node_backend(self, node: BaseSchedulerNode) -> BaseScheduling:
        # [FUNCTION DOC] Returns the CUTLASS scheduling if the node is a CUTLASS template, otherwise the Triton scheduling.
        if self._cutlass_scheduling.is_cutlass_template(node):
            return self._cutlass_scheduling
        return self._triton_scheduling

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Decide vertical fusion: True if CUTLASS allows it; False if either node is a CUTLASS template
        # otherwise; else defer to Triton.
        if self._cutlass_scheduling.can_fuse_vertical(node1, node2):
            return True
        elif self._cutlass_scheduling.is_cutlass_template(
            node1
        ) or self._cutlass_scheduling.is_cutlass_template(node2):
            return False
        return self._triton_scheduling.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Decide horizontal fusion: if either node is a CUTLASS template, defer to CUTLASS (currently always
        # False); otherwise defer to Triton.
        for node in (node1, node2):
            if self._cutlass_scheduling.is_cutlass_template(node):
                return self._cutlass_scheduling.can_fuse_horizontal(
                    node1, node2
                )  # always False at the moment
        return self._triton_scheduling.can_fuse_horizontal(node1, node2)

    def group_fn(
        self, sizes: Sequence[Sequence[_IntLike]]
    ) -> tuple[tuple[_IntLike, ...], ...]:
        # [FUNCTION DOC] Delegates iteration-size grouping to the Triton scheduling; returns the grouped sizes.
        return self._triton_scheduling.group_fn(sizes)

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> str | None:
        # [FUNCTION DOC] Generate code for a template node. CUTLASS templates are delegated to CUTLASS (raises AssertionError if
        # prologue nodes are present); other templates go to Triton. Returns the generated source or None.
        if self._cutlass_scheduling.is_cutlass_template(template_node):
            if prologue_nodes:
                raise AssertionError("cutlass template does not support prologue_nodes")
            return self._cutlass_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        else:
            return self._triton_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )

    def codegen_mix_order_reduction(self, node):
        # [FUNCTION DOC] Delegates mix-order reduction codegen for the node to Triton.
        return self._triton_scheduling.codegen_mix_order_reduction(node)

    def codegen_nested_reduction(self, node):
        # [FUNCTION DOC] Delegates nested reduction codegen for the node to Triton.
        return self._triton_scheduling.codegen_nested_reduction(node)

    def codegen_node(self, node: FusedSchedulerNode | SchedulerNode) -> None:
        # [FUNCTION DOC] Delegates codegen of a scheduler node to Triton.
        return self._triton_scheduling.codegen_node(node)

    def codegen_sync(self) -> None:
        # [FUNCTION DOC] Delegates synchronization codegen to Triton.
        return self._triton_scheduling.codegen_sync()

    def flush(self) -> None:
        # [FUNCTION DOC] Delegates flushing of pending kernels to Triton.
        return self._triton_scheduling.flush()

    def codegen_combo_kernel(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Delegates combo kernel codegen (all arguments passed through) to Triton.
        return self._triton_scheduling.codegen_combo_kernel(*args, **kwargs)

    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[float, str]:
        # [FUNCTION DOC] Delegates benchmarking of fused nodes to Triton; returns a timing and a source path string.
        return self._triton_scheduling.benchmark_fused_nodes(nodes)

    def benchmark_codegened_module(self, module):
        # [FUNCTION DOC] Delegates benchmarking of a generated module to Triton.
        return self._triton_scheduling.benchmark_codegened_module(module)

    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence[Any],
        benchmark_kernel: bool = False,
        hint_override: int | None = None,
    ) -> str:
        # [FUNCTION DOC] Delegates kernel source generation for the nodes to Triton, passing the benchmark flag and hint override.
        return self._triton_scheduling.generate_kernel_code_from_nodes(
            nodes, benchmark_kernel, hint_override=hint_override
        )

    def benchmark_combo_kernel(
        self, node_list: Sequence[BaseSchedulerNode], node_benchmark_results
    ) -> tuple[float, float, list[str | None]]:
        # [FUNCTION DOC] Delegates combo kernel benchmarking to Triton; returns timings and per-node source info.
        return self._triton_scheduling.benchmark_combo_kernel(
            node_list, node_benchmark_results
        )
