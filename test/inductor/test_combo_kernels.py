# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor combo kernels (horizontal fusion of independent Triton
# kernels). Covers the debug logging of partition separation in the combo kernel partitioner,
# and GPU end-to-end tests that compile small functions and check generated kernel counts,
# generated Triton source (via FileCheck), profiler grid sizes, autotuning, mutation,
# reductions, dynamic shapes and per-subkernel block sizes. Test classes are run both with and
# without the per-subkernel-blocks option via subclasses. Import-time side effect: tries to
# import check_model helpers from test_torchinductor and exits quietly (or re-raises) if that
# import is skipped.

import contextlib
import json
import logging
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch._inductor
from torch._dynamo.utils import counters
from torch._inductor.codegen.simd import NodeInfo
from torch._inductor.codegen.triton_combo_kernel import (
    _default_custom_combo_kernel_horizontal_partition,
    _log_partition_separation_once,
    LARGE_NUMELS,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import clear_caches, fresh_cache, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
    skipIfXpu,
    TestCase,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU_AND_TRITON
from torch.testing._internal.triton_utils import (
    requires_cuda_and_triton,
    requires_gpu_and_triton,
    requires_xpu_and_triton,
)


aten = torch.ops.aten

try:
    try:
        from .test_torchinductor import check_model, check_model_gpu
    except ImportError:
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            check_model,
            check_model_gpu,
        )
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


class _Features:
    # [CLASS DOC] HELPER: Minimal stand-in for a node features object. It stores a single flag
    # set at construction and reports it through is_reduction; used only to build fake
    # NodeInfo values.
    def __init__(self, is_reduction=False):
        # [FUNCTION DOC] HELPER: Store the is_reduction flag (default false) on the instance.
        self._is_reduction = is_reduction

    def is_reduction(self):
        # [FUNCTION DOC] HELPER: Return the stored reduction flag.
        return self._is_reduction


class _PartitionLoggingNode:
    # [CLASS DOC] HELPER: Fake scheduler node for partition tests. Its group attribute is a
    # one-item list holding a pair of (no sizes, reduction numel), which is all the partitioner
    # reads here.
    def __init__(self, rnumel=1):
        # [FUNCTION DOC] HELPER: Build the fake node with group holding None and the
        # given rnumel.
        self.group = [(None, rnumel)]


def _optimization_hint(expr, fallback=None):
    # [FUNCTION DOC] HELPER: Stand-in for the size-variable optimization hint: returns the
    # expression unchanged and ignores the fallback argument.
    return expr


def _make_node_info(x_hint, rnumel=1, is_reduction=False):
    # [FUNCTION DOC] HELPER: Build a NodeInfo with an empty node schedule, tiling of x equal to
    # x_hint and y equal to rnumel, numel equal to x_hint, the given rnumel, a fake features
    # object carrying is_reduction, and persistent reduction disabled.
    return NodeInfo(
        node_schedule=[],
        tiling={"x": x_hint, "y": rnumel},
        tiling_scores=None,
        numel=x_hint,
        rnumel=rnumel,
        features=_Features(is_reduction),
        is_persistent_reduction=False,
    )


class ComboKernelPartitionLoggingTests(TestCase):
    # [CLASS DOC] Tests that the combo kernel horizontal partitioner emits its "nodes are
    # separated" debug messages once per distinct partition context (not once per equivalent
    # recompile), and that the de-duplication cache resets when Inductor caches are cleared.
    def test_large_pointwise_separation_debug_log_uses_companion_context(self):
        # [FUNCTION DOC] Large pointwise node (x above the large-numel threshold) plus a small
        # companion is partitioned into two singleton groups, large first. Partitions three
        # times: twice with equivalent node infos and once with a differently sized companion.
        # Asserts the partition results and that the "large pointwise nodes are separated"
        # debug message appears exactly twice (the equivalent repeat is de-duplicated, the
        # different companion logs again). The de-dup cache is cleared before and after.
        large_x = int(LARGE_NUMELS) + 1
        large_node = _PartitionLoggingNode()
        small_node = _PartitionLoggingNode()
        equivalent_large_node = _PartitionLoggingNode()
        equivalent_small_node = _PartitionLoggingNode()
        different_companion_large_node = _PartitionLoggingNode()
        different_small_node = _PartitionLoggingNode()
        node_info_map = {
            large_node: _make_node_info(large_x),
            small_node: _make_node_info(1024),
            equivalent_large_node: _make_node_info(large_x),
            equivalent_small_node: _make_node_info(1024),
            different_companion_large_node: _make_node_info(large_x),
            different_small_node: _make_node_info(2048),
        }
        graph = SimpleNamespace(
            sizevars=SimpleNamespace(optimization_hint=_optimization_hint)
        )
        logger = logging.getLogger("torch._inductor.codegen.triton_combo_kernel")

        _log_partition_separation_once.cache_clear()
        try:
            with (
                V.set_graph_handler(graph),
                self.assertLogs(logger, level=logging.DEBUG) as cm,
            ):
                first_partitions = _default_custom_combo_kernel_horizontal_partition(
                    [large_node, small_node], None, node_info_map
                )
                second_partitions = _default_custom_combo_kernel_horizontal_partition(
                    [equivalent_large_node, equivalent_small_node],
                    None,
                    node_info_map,
                )
                third_partitions = _default_custom_combo_kernel_horizontal_partition(
                    [different_companion_large_node, different_small_node],
                    None,
                    node_info_map,
                )
        finally:
            _log_partition_separation_once.cache_clear()

        self.assertEqual(first_partitions, [[large_node], [small_node]])
        self.assertEqual(
            second_partitions, [[equivalent_large_node], [equivalent_small_node]]
        )
        self.assertEqual(
            third_partitions, [[different_companion_large_node], [different_small_node]]
        )
        large_pointwise_logs = [
            msg
            for msg in cm.output
            if "ComboKernels: 1 large pointwise nodes are separated" in msg
        ]
        self.assertEqual(len(large_pointwise_logs), 2)

    def test_long_reduction_separation_debug_log_uses_companion_context(self):
        # [FUNCTION DOC] Same scenario for long reductions (reduction numel 4096 vs 1024): the
        # short reduction is ordered before the long one in the partitions. Two equivalent runs
        # plus one with a different short companion (rnumel 1536) must produce exactly two
        # "long reduction nodes are separated" debug messages.
        long_node = _PartitionLoggingNode(rnumel=4096)
        short_node = _PartitionLoggingNode(rnumel=1024)
        equivalent_long_node = _PartitionLoggingNode(rnumel=4096)
        equivalent_short_node = _PartitionLoggingNode(rnumel=1024)
        different_companion_long_node = _PartitionLoggingNode(rnumel=4096)
        different_short_node = _PartitionLoggingNode(rnumel=1536)
        node_info_map = {
            long_node: _make_node_info(1024, rnumel=4096, is_reduction=True),
            short_node: _make_node_info(1024, rnumel=1024, is_reduction=True),
            equivalent_long_node: _make_node_info(1024, rnumel=4096, is_reduction=True),
            equivalent_short_node: _make_node_info(
                1024, rnumel=1024, is_reduction=True
            ),
            different_companion_long_node: _make_node_info(
                1024, rnumel=4096, is_reduction=True
            ),
            different_short_node: _make_node_info(1024, rnumel=1536, is_reduction=True),
        }
        graph = SimpleNamespace(
            sizevars=SimpleNamespace(optimization_hint=_optimization_hint)
        )
        logger = logging.getLogger("torch._inductor.codegen.triton_combo_kernel")

        _log_partition_separation_once.cache_clear()
        try:
            with (
                V.set_graph_handler(graph),
                self.assertLogs(logger, level=logging.DEBUG) as cm,
            ):
                first_partitions = _default_custom_combo_kernel_horizontal_partition(
                    [long_node, short_node], None, node_info_map
                )
                second_partitions = _default_custom_combo_kernel_horizontal_partition(
                    [equivalent_long_node, equivalent_short_node],
                    None,
                    node_info_map,
                )
                third_partitions = _default_custom_combo_kernel_horizontal_partition(
                    [different_companion_long_node, different_short_node],
                    None,
                    node_info_map,
                )
        finally:
            _log_partition_separation_once.cache_clear()

        self.assertEqual(first_partitions, [[short_node], [long_node]])
        self.assertEqual(
            second_partitions, [[equivalent_short_node], [equivalent_long_node]]
        )
        self.assertEqual(
            third_partitions,
            [[different_short_node], [different_companion_long_node]],
        )
        long_reduction_logs = [
            msg
            for msg in cm.output
            if "ComboKernels: 1 long reduction nodes are separated" in msg
        ]
        self.assertEqual(len(long_reduction_logs), 2)

    def test_long_reduction_log_with_very_large_reduction_split(self):
        # [FUNCTION DOC] Partition a long reduction, a reduction whose x size equals the
        # large-numel threshold, and a short reduction. Expects the order very-large, short,
        # long, and exactly one "long reduction" message plus exactly one "very large
        # reduction" message.
        long_node = _PartitionLoggingNode(rnumel=4096)
        very_large_node = _PartitionLoggingNode(rnumel=4096)
        short_node = _PartitionLoggingNode(rnumel=1024)
        node_info_map = {
            long_node: _make_node_info(1024, rnumel=4096, is_reduction=True),
            very_large_node: _make_node_info(
                int(LARGE_NUMELS), rnumel=4096, is_reduction=True
            ),
            short_node: _make_node_info(1024, rnumel=1024, is_reduction=True),
        }
        graph = SimpleNamespace(
            sizevars=SimpleNamespace(optimization_hint=_optimization_hint)
        )
        logger = logging.getLogger("torch._inductor.codegen.triton_combo_kernel")

        _log_partition_separation_once.cache_clear()
        try:
            with (
                V.set_graph_handler(graph),
                self.assertLogs(logger, level=logging.DEBUG) as cm,
            ):
                partitions = _default_custom_combo_kernel_horizontal_partition(
                    [long_node, very_large_node, short_node], None, node_info_map
                )
        finally:
            _log_partition_separation_once.cache_clear()

        self.assertEqual(partitions, [[very_large_node], [short_node], [long_node]])
        self.assertEqual(
            len(
                [
                    msg
                    for msg in cm.output
                    if "ComboKernels: 1 long reduction nodes are separated" in msg
                ]
            ),
            1,
        )
        self.assertEqual(
            len(
                [
                    msg
                    for msg in cm.output
                    if "ComboKernels: 1 very large reduction nodes are separated" in msg
                ]
            ),
            1,
        )

    def test_partition_separation_debug_log_cache_clears_with_inductor_caches(self):
        # [FUNCTION DOC] Verifies the de-dup cache is dropped by clear_caches: partition two
        # equivalent large/small pairs, clear Inductor caches, then partition a third
        # equivalent pair. The large pointwise separation message must be logged twice (before
        # the clear and after it).
        large_x = int(LARGE_NUMELS) + 1
        large_node = _PartitionLoggingNode()
        small_node = _PartitionLoggingNode()
        equivalent_large_node = _PartitionLoggingNode()
        equivalent_small_node = _PartitionLoggingNode()
        after_clear_large_node = _PartitionLoggingNode()
        after_clear_small_node = _PartitionLoggingNode()
        node_info_map = {
            large_node: _make_node_info(large_x),
            small_node: _make_node_info(1024),
            equivalent_large_node: _make_node_info(large_x),
            equivalent_small_node: _make_node_info(1024),
            after_clear_large_node: _make_node_info(large_x),
            after_clear_small_node: _make_node_info(1024),
        }
        graph = SimpleNamespace(
            sizevars=SimpleNamespace(optimization_hint=_optimization_hint)
        )
        logger = logging.getLogger("torch._inductor.codegen.triton_combo_kernel")

        _log_partition_separation_once.cache_clear()
        try:
            with (
                V.set_graph_handler(graph),
                self.assertLogs(logger, level=logging.DEBUG) as cm,
            ):
                _default_custom_combo_kernel_horizontal_partition(
                    [large_node, small_node], None, node_info_map
                )
                _default_custom_combo_kernel_horizontal_partition(
                    [equivalent_large_node, equivalent_small_node],
                    None,
                    node_info_map,
                )
                clear_caches()
                _default_custom_combo_kernel_horizontal_partition(
                    [after_clear_large_node, after_clear_small_node],
                    None,
                    node_info_map,
                )
        finally:
            _log_partition_separation_once.cache_clear()

        large_pointwise_logs = [
            msg
            for msg in cm.output
            if "ComboKernels: 1 large pointwise nodes are separated" in msg
        ]
        self.assertEqual(len(large_pointwise_logs), 2)


@instantiate_parametrized_tests
class ComboKernelTests(TestCase):
    # [CLASS DOC] End-to-end combo kernel tests on GPU with Triton. setUp enables combo kernels
    # with benchmarking off, unlimited distance, no peak-memory limits, and per-subkernel
    # blocks taken from the class attribute (false here; subclasses flip it). Also holds the
    # check_model helpers and the flag asking for kernel count checks. Each test compiles a
    # function and asserts results match eager and typically the number of generated kernels or
    # substrings of the generated code.
    check_model_gpu = check_model_gpu
    check_model_cpu = check_model
    check_kernel_count = True
    combo_kernel_per_subkernel_blocks = False

    def setUp(self):
        # [FUNCTION DOC] Reset Inductor metrics, then open an ExitStack holding an Inductor
        # config patch that enables combo kernels (benchmark off, max distance -1, no
        # memory-increase limits, and the per-subkernel-blocks option from the class
        # attribute). Undone in tearDown.
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                    "combo_kernel_per_subkernel_blocks": self.combo_kernel_per_subkernel_blocks,
                    "combo_kernel_max_distance": -1,
                    "combo_kernel_peak_memory_increase_gb": None,
                    "combo_kernel_peak_memory_pct_threshold": None,
                }
            )
        )

    def tearDown(self):
        # [FUNCTION DOC] Close the config patch stack, reset Inductor metrics, then call the
        # base tearDown.
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    def test_activation_functions(self):
        # [FUNCTION DOC] Relu, sigmoid and tanh on differently shaped inputs must give
        # eager-equal results and fuse into a single generated kernel.
        def test_activations(a, b, c):
            # [FUNCTION DOC] HELPER: Apply relu, sigmoid and tanh to three inputs and return
            # the three results.
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    def test_reduce_functions(self):
        # [FUNCTION DOC] Sum, max and min reductions along dim 0 plus a tanh pointwise op:
        # compiled output matches eager and at most two kernels are generated.
        def test_reduce(a, b, c, d):
            # [FUNCTION DOC] HELPER: Return sum, max (values and indices) and min results plus
            # tanh of the last.
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(torch._inductor.metrics.generated_kernel_count <= 2)

    @requires_gpu_and_triton
    def test_mutated_args(self):
        # [FUNCTION DOC] In-place add, in-place sigmoid and in-place tanh on inputs plus one
        # out-of-place add: compiled outputs equal eager and everything fuses into one kernel.
        def test_mutated(a, b, c, d):
            # [FUNCTION DOC] HELPER: Mutate three inputs in place, add 5 out of place to the
            # third, return all.
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    def test_reduce_split(self):
        # [FUNCTION DOC] A full vector norm of a large matrix (needs a split reduction)
        # together with a small column sum must still compile and match eager results.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return the vector norm of the first input and the dim 0
            # sum of the second.
            a1 = torch.linalg.vector_norm(a)
            b1 = torch.sum(b, dim=0)
            return a1, b1

        inps = [
            torch.rand(2048, 512, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

    @requires_gpu_and_triton
    def test_2d_blocking_partitioning(self):
        # [FUNCTION DOC] Three adds where two operands are transposed (2D tiling with
        # blocking) must yield two kernels. When cpp_wrapper and per-subkernel blocks are both
        # on, also profiles two runs, exports a chrome trace, finds the fused pointwise combo
        # kernel events, and asserts their grid is smaller than the degenerate
        # one-element-per-block grid, guarding against missing per-subkernel block meta keys
        # in the wrapper grid computation.
        def fn(a0, a1, a2, b0, b1, b2):
            # [FUNCTION DOC] HELPER: Add three pairs of tensors and return the three sums.
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        inps = (
            torch.rand(30, 20, device=GPU_TYPE),
            torch.rand(40, 30, device=GPU_TYPE),
            torch.rand(36, 40, device=GPU_TYPE),
            torch.rand(30, 20, device=GPU_TYPE),
            torch.rand(30, 40, device=GPU_TYPE).t(),
            torch.rand(40, 36, device=GPU_TYPE).t(),
        )
        self.check_model_gpu(fn, inps)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

        # Verify the cpp_wrapper grid computation uses per-subkernel block sizes.
        # Without per-subkernel block support, generate_lazy only provides shared
        # meta keys (XBLOCK, YBLOCK) but SequentialFlattenComboKernelGrid looks
        # up XBLOCK_0, XBLOCK_1 etc. — dict.get returns None, and ceildiv treats
        # None as block=1, hardcoding the grid to xnumel*ynumel per subkernel.
        if (
            torch._inductor.config.cpp_wrapper
            and self.combo_kernel_per_subkernel_blocks
        ):
            from torch.profiler import ProfilerActivity

            fn_c = torch.compile(fn)
            expected = fn(*inps)
            activity = getattr(ProfilerActivity, GPU_TYPE.upper())
            with tempfile.NamedTemporaryFile(suffix=".json") as trace_file:
                with torch.profiler.profile(activities=[activity]) as prof:
                    actual = fn_c(*inps)
                    actual = fn_c(*inps)
                self.assertEqual(expected, actual)

                prof.export_chrome_trace(trace_file.name)
                with open(trace_file.name) as f:
                    trace_json = json.load(f)

            combo_events = [
                e
                for e in trace_json["traceEvents"]
                if "triton_poi_fused_1" in e.get("name", "")
            ]
            self.assertTrue(len(combo_events) > 0)
            # With proper block sizes (>=16), the grid should be much smaller
            # than the degenerate 30*40 + 40*36 = 2640 (block=1).
            degenerate_grid = 30 * 40 + 40 * 36
            for e in combo_events:
                grid = e.get("args", {}).get("grid")
                if grid:
                    self.assertLess(grid[0], degenerate_grid)

    @requires_gpu_and_triton
    def test_persistent_reduction_size_hint(self):
        # [FUNCTION DOC] Two row max/min reductions over different widths must compile to a
        # persistent reduction kernel whose generated source has the expected size hints;
        # output matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return max over dim 1 of x and min over dim 1 of y.
            return x.max(1), y.min(1)

        inps = (
            torch.rand(768, 16, device=GPU_TYPE),
            torch.rand(768, 32, device=GPU_TYPE),
        )

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        FileCheck().check("triton_heuristics.persistent_reduction").check(
            "size_hints={'x': 1024, 'r0_': 32}"
        ).run(code[0])
        self.assertEqual(out_eager, out_compiled)

    @requires_cuda_and_triton
    @unittest.skipIf(
        not SM90OrLater or torch.version.hip,
        "TMA requires SM90 or later (Hopper+), not supported on ROCm",
    )
    @torch._inductor.config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
        }
    )
    def test_combo_kernel_tma_descriptor_block_name(self):
        # [FUNCTION DOC] Needs CUDA, Hopper or newer and non-ROCm; runs with tensor descriptor
        # use and aligned inputs. Two row sums of different widths must match eager and the
        # generated code must reference tensor descriptor creation (checks the combo kernel
        # block naming works with TMA).
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return last-dim sums of both inputs.
            return a.sum(dim=-1), b.sum(dim=-1)

        inps = [
            torch.randn(1024, 128, device=GPU_TYPE),
            torch.randn(1024, 256, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        out_compiled, code = run_and_get_code(torch.compile(fn), *inps)
        torch.testing.assert_close(out_eager, out_compiled, atol=1e-4, rtol=1e-4)
        FileCheck().check("make_tensor_descriptor").run(code[0])

    @requires_gpu_and_triton
    def test_sort_in_combo_kernel_forces_persistent_reduction(self):
        # [FUNCTION DOC] Sort sub-kernels in a combo kernel must be forced to persistent
        # reduction even if the choices heuristic says otherwise. A patched choices method
        # returns false for kernels containing sort, while config enables per-subkernel blocks,
        # mixed sizes and combo autotuning. Compiling must not fail and the output must match
        # eager and use a persistent reduction kernel.
        # ops.sort only works under persistent reduction. Combo kernel codegen
        # must override the persistent_reduction heuristic for sort sub-kernels,
        # otherwise the TritonKernel.sort() assertion fires.
        from torch._inductor.choices import InductorChoices

        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Sort three integer inputs along dim 1 and return values;
            # tanh of the fourth.
            a1 = torch.sort(a, dim=1).values
            b1 = torch.sort(b, dim=1).values
            c1 = torch.sort(c, dim=1).values
            d1 = torch.nn.functional.tanh(d)
            return a1, b1, c1, d1

        inps = [
            torch.randint(0, 1000, (10, 200), device=GPU_TYPE),
            torch.randint(0, 1000, (20, 200), device=GPU_TYPE),
            torch.randint(0, 1000, (10, 200), device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        orig = InductorChoices.should_use_persistent_reduction

        @staticmethod
        def force_non_persistent_for_sort(features, cooperative_reduction):
            # [FUNCTION DOC] HELPER: Replacement for the persistent-reduction heuristic:
            # returns false for features containing a sort op, otherwise defers to the saved
            # original implementation.
            if features.contains_op("sort"):
                return False
            return orig(features, cooperative_reduction)

        out_eager = fn(*inps)
        with (
            patch.object(
                InductorChoices,
                "should_use_persistent_reduction",
                force_non_persistent_for_sort,
            ),
            torch._inductor.config.patch(
                {
                    "combo_kernel_per_subkernel_blocks": True,
                    "combo_kernel_max_num_nodes": 128,
                    "combo_kernel_allow_mixed_sizes": 2,
                    "combo_kernels_autotune": 2,
                    "combo_kernel_autotune_grouping": True,
                }
            ),
        ):
            fn_c = torch.compile(fn)
            out_compiled, code = run_and_get_code(fn_c, *inps)
        self.assertEqual(out_eager, out_compiled)
        FileCheck().check("triton_heuristics.persistent_reduction").run(code[0])

    @requires_gpu_and_triton
    def test_fuse_mix_order_reductions_combo_kernels(self):
        # [FUNCTION DOC] Mix-order reductions (row and column sums of the same input) are fused
        # into a separate kernel, a consumer of the row sum is excluded from combos, and two
        # independent scalar sums become a combo kernel: expects output equal to eager and
        # three generated kernels total.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Return the sum of the row sums, the column sums, and two
            # scalar sums.
            # FusedMixOrderReductions produces row_sum (buf0)
            row_sum = x.sum(dim=1)
            col_sum = x.sum(dim=0)

            # consumer of row_sum - excluded from combo kernels
            row_sum_reduced = row_sum.sum()  # reads buf0

            # independent reductions - combo-kerneled
            y_sum = y.sum()
            z_sum = z.sum()

            return row_sum_reduced, col_sum, y_sum, z_sum

        inps = [
            torch.rand(8192, 1024, device=GPU_TYPE),
            torch.rand(2048, device=GPU_TYPE),
            torch.rand(2048, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        self.assertEqual(out_eager, out_compiled)
        # [row_sum, col_sum] will became 1 kernel MixOrderReductionGrid
        # [row_sum_reduced] will become a separate kernel due to the consumer
        # [y_sum, z_sum] will become a combo kernel
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 3)

    @requires_gpu_and_triton
    def test_combo_kernel_scalar_store_broadcast(self):
        # [FUNCTION DOC] Scalar add, a row sum and a scalar reduction depending on the
        # scalar add are combined: results close to eager (loose tolerance) and exactly one
        # kernel generated.
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Compute a scalar add, a row sum, and a full sum of a
            # product with it.
            scalar_sum = a + b
            vector_result = c.sum(dim=1)
            scalar_red = (d * scalar_sum).sum()
            return scalar_sum, vector_result, scalar_red

        inps = [
            torch.tensor(1.0, device=GPU_TYPE),
            torch.tensor(2.0, device=GPU_TYPE),
            torch.randn(2048, 2048, device=GPU_TYPE),
            torch.randn(2048, 1, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        torch.testing.assert_close(out_eager, out_compiled, rtol=1e-4, atol=1e-4)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    @parametrize("benchmark_combo_kernel", [False, True])
    def test_single_combo_kernels(self, benchmark_combo_kernel):
        # [FUNCTION DOC] Parametrized by benchmark_combo_kernel. A big and a small tensor each
        # get a pointwise add; since only single-node groups exist, no combo kernel is
        # produced. Checks output, kernel count (four with benchmarking, two otherwise) and
        # that generated code is a plain 1D pointwise kernel without combo grid meta.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Add 1 to each input and return both.
            c = torch.add(a, 1)
            d = torch.add(b, 1)
            return c, d

        inps = [
            torch.rand(8192, 8192, device=GPU_TYPE),
            torch.rand(100, 100, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        with torch._inductor.config.patch(
            "benchmark_combo_kernel", benchmark_combo_kernel
        ):
            fn_c = torch.compile(fn)
            out_compiled, code = run_and_get_code(fn_c, *inps)
            self.assertEqual(out_eager, out_compiled)
            # With benchmark_combo_kernel=True, kernel count is 2x due to benchmarking
            # (1x benchmarking + 1x actual codegen, single-node code generation skipped)
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count,
                4 if benchmark_combo_kernel else 2,
            )
            # Verify kernels are regular pointwise, not combo kernels
            FileCheck().check("triton_heuristics.pointwise").check(
                "'grid_type': 'Grid1D'"
            ).check_not("combo_grid_meta").run(code[0])

    @requires_gpu_and_triton
    def test_combo_kernel_per_config_subkernel_poi(self):
        # [FUNCTION DOC] Two pointwise ops of very different sizes make one kernel. With
        # per-subkernel blocks on, the code must declare separate block parameters per
        # subkernel; otherwise none.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return the first input times 2 and the second plus 1.
            o1 = a * 2.0
            o2 = b + 1.0
            return o1, o2

        inps = [
            torch.randn(512, device=GPU_TYPE),
            torch.randn(524288, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
        if torch._inductor.config.combo_kernel_per_subkernel_blocks:
            FileCheck().check("XBLOCK_0 : tl.constexpr").check(
                "XBLOCK_1 : tl.constexpr"
            ).run(code[0])
        else:
            FileCheck().check_not("XBLOCK_0 : tl.constexpr").run(code[0])

    @requires_gpu_and_triton
    def test_combo_kernel_per_config_subkernel_per(self):
        # [FUNCTION DOC] Two row sums of different widths give one kernel. With per-subkernel
        # blocks the generated code has per-subkernel reduction block constants (64 and 512);
        # without, a shared block.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return last-dim sums of both inputs.
            return a.sum(dim=-1), b.sum(dim=-1)

        inps = [
            torch.randn(1024, 64, device=GPU_TYPE),
            torch.randn(1024, 512, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        torch.testing.assert_close(out_eager, out_compiled, atol=1e-4, rtol=1e-4)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
        if torch._inductor.config.combo_kernel_per_subkernel_blocks:
            FileCheck().check("R0_BLOCK: tl.constexpr = 64").check(
                "R0_BLOCK: tl.constexpr = 512"
            ).run(code[0])
        else:
            FileCheck().check("XBLOCK : tl.constexpr").check_not(
                "XBLOCK_0 : tl.constexpr"
            ).run(code[0])

    @requires_gpu_and_triton
    def test_combo_kernel_per_config_subkernel_red_per(self):
        # [FUNCTION DOC] Reductions with persistent (width 128) and regular (width 2048) styles
        # combine into one kernel; per-subkernel blocks produce separate block parameters for
        # the second.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return the last-dim sums of both inputs.
            r1 = a.sum(dim=-1)
            r2 = b.sum(dim=-1)
            return r1, r2

        inps = [
            torch.randn(512, 128, device=GPU_TYPE),  # Persistent (r0=128)
            torch.randn(256, 2048, device=GPU_TYPE),  # Regular (r0=2048)
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        torch.testing.assert_close(out_eager, out_compiled, atol=1e-4, rtol=1e-4)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
        if torch._inductor.config.combo_kernel_per_subkernel_blocks:
            FileCheck().check("XBLOCK_0 : tl.constexpr").check(
                "XBLOCK_1 : tl.constexpr, R0_BLOCK_1 : tl.constexpr"
            ).run(code[0])
        else:
            FileCheck().check_not("XBLOCK_0 : tl.constexpr").run(code[0])

    @requires_gpu_and_triton
    def test_combo_kernel_per_config_subkernel_red(self):
        # [FUNCTION DOC] Two reductions over dims 0 and 2 of 3D inputs combine into one
        # kernel; with per-subkernel blocks both subkernels declare their own X and reduction
        # block parameters.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return sums over dims 0 and 2 of both inputs.
            r1 = a.sum(dim=(0, 2))
            r2 = b.sum(dim=(0, 2))
            return r1, r2

        inps = [
            torch.randn(32, 64, 128, device=GPU_TYPE),
            torch.randn(32, 64, 128, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        torch.testing.assert_close(out_eager, out_compiled, atol=1e-4, rtol=1e-4)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
        if torch._inductor.config.combo_kernel_per_subkernel_blocks:
            FileCheck().check(
                " XBLOCK_0 : tl.constexpr, R0_BLOCK_0 : tl.constexpr"
            ).check("XBLOCK_1 : tl.constexpr, R0_BLOCK_1 : tl.constexpr").run(code[0])
        else:
            FileCheck().check("XBLOCK : tl.constexpr").check(
                "R0_BLOCK : tl.constexpr"
            ).check_not("XBLOCK_0 : tl.constexpr").run(code[0])

    @skipIfRocm
    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        {
            "triton.prefer_nd_tiling": True,
            "triton.tile_reductions": True,
        }
    )
    def test_combo_kernel_per_config_subkernel_r0_r1(self):
        # [FUNCTION DOC] Skipped on ROCm. With ND tiling and tiled reductions, 2D
        # reductions over two dims on permuted inputs are separated from combos, so two
        # kernels are generated.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return sums over dims 1 and 2 of both inputs.
            return a.sum(dim=(1, 2)), b.sum(dim=(1, 2))

        inps = [
            torch.randn(32, 16, 64, device=GPU_TYPE).permute(1, 0, 2),
            torch.randn(32, 16, 64, device=GPU_TYPE).permute(1, 0, 2),
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        self.assertEqual(out_eager, out_compiled)
        # 2D reduction kernels (r0_, r1_) are separated from combo kernels
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        {
            "triton.prefer_nd_tiling": True,
            "triton.max_tiles": 3,
        }
    )
    def test_combo_kernel_per_config_subkernel_poi_3d(self):
        # [FUNCTION DOC] With ND tiling and up to three tiles, 3D pointwise ops on sliced views
        # are separated from combo kernels: output equals eager and two kernels are generated.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return the first input plus 1 and the second times 2.
            return a + 1.0, b * 2.0

        inps = [
            torch.randn(16, 16, 16, device=GPU_TYPE)[:8, :8, :8],
            torch.randn(16, 16, 16, device=GPU_TYPE)[:8, :8, :8],
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        self.assertEqual(out_eager, out_compiled)
        # 3D poi (x, y, z) are separated from combo kernels
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @skipIfXpu(msg="Profiler JSON traceEvents is not supported on XPU")
    @requires_gpu_and_triton
    def test_combo_kernel_per_config_subkernel_block_size(self):
        # [FUNCTION DOC] Skipped on XPU. Contiguous channels-last conversions of eight
        # differently shaped tensors fuse into one combo kernel. Profiles the run, finds the
        # fused kernel event and checks its grid (single dimension with per-subkernel blocks,
        # else a 2D grid), then checks results, kernel count and the pid offset code form.
        from torch.profiler import ProfilerActivity

        def fn(t0, t1, t2, t3, t4, t5, t6, t7):
            # [FUNCTION DOC] HELPER: Convert each of eight inputs to channels-last layout and
            # return them.
            o0 = t0.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=12,    xnumel=50176
            o1 = t1.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=192,   xnumel=9
            o2 = t2.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=4096,  xnumel=9
            o3 = t3.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=8192,  xnumel=9
            o4 = t4.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=16384, xnumel=9
            o5 = t5.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=32768, xnumel=9
            o6 = t6.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=65536, xnumel=9
            o7 = t7.contiguous(
                memory_format=torch.channels_last
            )  # ynumel=65536, xnumel=9
            return o0, o1, o2, o3, o4, o5, o6, o7

        inps = [
            torch.randn(4, 3, 224, 224, device=GPU_TYPE),
            torch.randn(64, 3, 3, 3, device=GPU_TYPE),
            torch.randn(64, 64, 3, 3, device=GPU_TYPE),
            torch.randn(128, 64, 3, 3, device=GPU_TYPE),
            torch.randn(128, 128, 3, 3, device=GPU_TYPE),
            torch.randn(256, 128, 3, 3, device=GPU_TYPE),
            torch.randn(256, 256, 3, 3, device=GPU_TYPE),
            torch.randn(256, 256, 3, 3, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        fn_c = torch.compile(fn)

        with tempfile.NamedTemporaryFile(suffix=".json") as trace_file:
            trace_path = trace_file.name
            activity = getattr(ProfilerActivity, GPU_TYPE.upper())

            with torch.profiler.profile(
                activities=[activity],
                record_shapes=True,
            ) as prof:
                out_compiled, code = run_and_get_code(fn_c, *inps)

            prof.export_chrome_trace(trace_path)

            with open(trace_path) as f:
                trace_json = json.load(f)

            triton_events = [
                event
                for event in trace_json["traceEvents"]
                if "triton_poi_fused_0" in event["name"]
            ]
            if torch._inductor.config.combo_kernel_per_subkernel_blocks:
                self.assertEqual([3795, 1, 1], triton_events[0]["args"]["grid"])
            else:
                self.assertEqual([791, 4096, 1], triton_events[0]["args"]["grid"])

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)
        if torch._inductor.config.combo_kernel_per_subkernel_blocks:
            FileCheck().check("x_pid_offset = local_pid % x_blocks_0").check(
                "y_pid_offset = local_pid // x_blocks_0"
            ).run(code[0])
        else:
            FileCheck().check("pid_offset = pid").run(code[0])

    @skipIfXpu(msg="Profiler JSON traceEvents is not supported on XPU")
    @requires_gpu_and_triton
    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_combo_kernel_dynamic_shapes_grid_changes(self):
        # [FUNCTION DOC] Skipped on XPU; dynamic shapes. Profiles the combo kernel for two
        # input shape sets and asserts the first grid dimension differs between them (grid
        # computed at runtime), that results match eager, and that with per-subkernel blocks
        # the second grid dimension is one.
        from torch.profiler import ProfilerActivity

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the sine of x and the cosine of y.
            return x.sin(), y.cos()

        fn_c = torch.compile(fn)

        def get_grid(x, y):
            # [FUNCTION DOC] HELPER: Run the compiled function under the profiler, export the
            # chrome trace to a temp file, and return the grid of the first fused pointwise
            # kernel event with the outputs.
            with tempfile.NamedTemporaryFile(suffix=".json") as trace_file:
                trace_path = trace_file.name
                activity = getattr(ProfilerActivity, GPU_TYPE.upper())
                with torch.profiler.profile(
                    activities=[activity],
                    record_shapes=True,
                ) as prof:
                    out = fn_c(x, y)
                prof.export_chrome_trace(trace_path)

                with open(trace_path) as f:
                    trace_json = json.load(f)
                triton_events = [
                    e
                    for e in trace_json["traceEvents"]
                    if "triton_poi_fused" in e["name"]
                ]
                return triton_events[0]["args"]["grid"], out

        x1 = torch.randn(1024, 512, device=GPU_TYPE)
        y1 = torch.randn(2048, 256, device=GPU_TYPE)
        grid1, out1 = get_grid(x1, y1)
        eager_out1 = fn(x1, y1)

        x2 = torch.randn(512, 256, device=GPU_TYPE)
        y2 = torch.randn(128, 64, device=GPU_TYPE)
        grid2, out2 = get_grid(x2, y2)
        eager_out2 = fn(x2, y2)

        self.assertNotEqual(grid1[0], grid2[0])
        self.assertEqual(out1, eager_out1)
        self.assertEqual(out2, eager_out2)

        if torch._inductor.config.combo_kernel_per_subkernel_blocks:
            self.assertEqual(grid1[1], 1)
            self.assertEqual(grid2[1], 1)

    @requires_gpu_and_triton
    @parametrize("pointwise_only,expected_kernel_count", [(False, 2), (True, 3)])
    def test_combo_kernels_pointwise_only(self, pointwise_only, expected_kernel_count):
        # [FUNCTION DOC] Parametrized by combo_kernels_pointwise_only: two pointwise ops and
        # two reductions produce 2 kernels normally and 3 when only pointwise nodes may be
        # combined. Checks eager equality.
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Return two pointwise results, a sum and a mean over the
            # last dim.
            p1 = a * 2.0
            p2 = b + 1.0
            r1 = c.sum(dim=-1)
            r2 = d.mean(dim=-1)
            return p1, p2, r1, r2

        inps = [
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(32, 1024, device=GPU_TYPE),
            torch.rand(32, 1024, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)

        torch._inductor.metrics.reset()
        with torch._inductor.config.patch(
            "combo_kernels_pointwise_only", pointwise_only
        ):
            fn_c = torch.compile(fn)
            out_compiled, _ = run_and_get_code(fn_c, *inps)
            self.assertEqual(out_eager, out_compiled)
            # With pointwise_only=True, we expect more kernels because reductions are not combined with pointwise ops
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count, expected_kernel_count
            )

    @requires_gpu_and_triton
    @parametrize(
        "max_num_nodes,expected_kernel_count",
        [(8, 1), (3, 2), (2, 3)],
    )
    def test_combo_kernel_max_num_nodes(self, max_num_nodes, expected_kernel_count):
        # [FUNCTION DOC] Parametrized by the combo_kernel_max_num_nodes limit (8, 3, 2): the
        # six pointwise ops yield 1, 2 or 3 kernels. Metrics are reset before compiling; output
        # must equal eager.
        def fn(a, b, c, d, e, f):
            # [FUNCTION DOC] HELPER: Return six different pointwise results of the six inputs.
            return (
                a * 2.0,
                b + 1.0,
                c.sin(),
                d.cos(),
                e.exp(),
                f.neg(),
            )

        inps = [
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)

        torch._inductor.metrics.reset()
        with torch._inductor.config.patch("combo_kernel_max_num_nodes", max_num_nodes):
            fn_c = torch.compile(fn)
            out_compiled, _ = run_and_get_code(fn_c, *inps)
            self.assertEqual(out_eager, out_compiled)
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count, expected_kernel_count
            )

    # waves_per_eu, matrix_instr_nonkdim, and kpack are HIP-only Triton
    # compile options, so only ROCm exercises this combo-kernel rewrite path.
    @unittest.skipIf(not torch.version.hip, "ROCm only")
    @requires_gpu_and_triton
    @parametrize("max_autotune", [False, True])
    def test_combo_kernel_amd_special_config_args(self, max_autotune):
        # [FUNCTION DOC] ROCm only, parametrized by max_autotune. Skips itself unless
        # per-subkernel blocks are enabled. Two pointwise ops compile with and without
        # max_autotune (HIP-only Triton options get rewritten in the combo path); output must
        # equal eager and one kernel is generated.
        if not torch._inductor.config.combo_kernel_per_subkernel_blocks:
            self.skipTest("requires combo_kernel_per_subkernel_blocks")

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return the first input times 2 and the second plus 1.
            return a * 2.0, b + 1.0

        inps = [
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)

        torch._inductor.metrics.reset()
        with torch._inductor.config.patch("max_autotune", max_autotune):
            fn_c = torch.compile(fn)
            out_compiled, _ = run_and_get_code(fn_c, *inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @skipIfXpu(msg="Profiler JSON traceEvents is not supported on XPU")
    @requires_gpu_and_triton
    @unittest.skipIf(not SM90OrLater, "Avoid oom on CI")
    def test_combo_kernel_yz_overflow(self):
        # [FUNCTION DOC] Skipped on XPU and before SM90. Two huge permute-clone-reshape
        # workloads are fused; the profiler trace of the fused kernel gives the grid: one
        # flattened dimension with per-subkernel blocks, otherwise a three dimensional grid
        # whose y extent is large. Checks the grid handles y/z size limits and that results
        # equal eager.
        from torch.profiler import ProfilerActivity

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Permute, clone contiguous and reshape each input to
            # two dimensions.
            a_permute = a.permute(0, 2, 1)
            a_clone = a_permute.clone(memory_format=torch.contiguous_format)
            a_view = a_clone.view(-1, a.shape[1])

            b_permute = b.permute(0, 2, 1)
            b_clone = b_permute.clone(memory_format=torch.contiguous_format)
            b_view = b_clone.view(-1, b.shape[1])
            return a_view, b_view

        inps = (
            torch.rand(4800, 34, 256, device=GPU_TYPE),
            torch.rand(22630, 44, 256, device=GPU_TYPE),
        )

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)

        with tempfile.NamedTemporaryFile(suffix=".json") as trace_file:
            trace_path = trace_file.name
            activity = getattr(ProfilerActivity, GPU_TYPE.upper())

            with torch.profiler.profile(
                activities=[activity],
                record_shapes=True,
            ) as prof:
                out_compiled, code = run_and_get_code(fn_c, *inps)

            prof.export_chrome_trace(trace_path)

            with open(trace_path) as f:
                trace_json = json.load(f)

            triton_events = [
                event
                for event in trace_json["traceEvents"]
                if "triton_poi_fused_0" in event["name"]
            ]

            if torch._inductor.config.combo_kernel_per_subkernel_blocks:
                self.assertEqual([83660, 1, 1], triton_events[0]["args"]["grid"])
            else:
                self.assertEqual([4, 45260, 2], triton_events[0]["args"]["grid"])

        self.assertEqual(out_eager, out_compiled)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        {
            "combo_kernels_autotune": 0,
            "combo_kernel_per_subkernel_blocks": True,
        }
    )
    def test_combo_kernel_no_bench_stitched_config(self):
        # [FUNCTION DOC] With autotuning off and per-subkernel blocks on, three pointwise ops
        # compile to exactly one Triton kernel (counted from async compile calls in the code).
        # The emitted code must contain a default config entry carrying XBLOCK values for
        # subkernels 0..2 and must not declare per-subkernel XBLOCK constants with default
        # values in the signature.
        import re

        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Return a times 2, b plus 1, c minus 1.
            return a * 2.0, b + 1.0, c - 1.0

        inps = [
            torch.rand(256, device=GPU_TYPE),
            torch.rand(256, device=GPU_TYPE),
            torch.rand(256, device=GPU_TYPE),
        ]
        out_eager = fn(*inps)
        out_compiled, code = run_and_get_code(torch.compile(fn), *inps)
        self.assertEqual(out_eager, out_compiled)
        combined = " ".join(code)
        # Count emitted kernels by their async_compile.triton(...) calls;
        # generated_kernel_count is inflated by throwaway probe codegen so
        # is not a reliable count of real kernels.
        self.assertEqual(combined.count("async_compile.triton("), 1)

        default_match = re.search(r"'default_config':\s*\{([^}]*)\}", combined)
        self.assertIsNotNone(default_match, "default_config not emitted")
        default_kvs = dict(
            re.findall(r"'(XBLOCK_\d+)':\s*(\d+)", default_match.group(1))
        )
        self.assertIsNone(re.search(r"XBLOCK_\d+:\s*tl\.constexpr\s*=", combined))
        self.assertEqual(set(default_kvs), {"XBLOCK_0", "XBLOCK_1", "XBLOCK_2"})

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        {
            "combo_kernels_autotune": 0,
            "combo_kernel_per_subkernel_blocks": True,
        }
    )
    def test_combo_kernel_no_bench_persistent_reduction(self):
        # [FUNCTION DOC] Autotuning off, per-subkernel blocks on, three row sums over 1024
        # elements. Output equals eager (looser tolerance on XPU because of summation order).
        # One Triton kernel; its default config holds only the XBLOCK entries for the three
        # subkernels, no reduction block.
        import re

        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Return the last-dim sum of each of the three inputs.
            return a.sum(-1), b.sum(-1), c.sum(-1)

        inps = [torch.randn(64, 1024, device=GPU_TYPE) for _ in range(3)]
        out_eager = fn(*inps)
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        out_compiled, code = run_and_get_code(torch.compile(fn), *inps)
        if GPU_TYPE == "xpu":
            # For this shape (64, 1024), XPU eager uses fewer threads (e.g. 32)
            # with strided access while Triton uses more threads (e.g. 256)
            # with contiguous access. Due to float32 non-associativity,
            # different element-to-thread mappings produce slightly different sums.
            self.assertEqual(out_eager, out_compiled, atol=2e-5, rtol=1e-5)
        else:
            self.assertEqual(out_eager, out_compiled)
        combined = " ".join(code)
        self.assertEqual(combined.count("async_compile.triton("), 1)

        default_match = re.search(r"'default_config':\s*\{([^}]*)\}", combined)
        self.assertIsNotNone(default_match)
        default_body = default_match.group(1)
        # default_config must only carry XBLOCK_i; R*_BLOCK_i is body-baked.
        self.assertNotIn("R0_BLOCK", default_body)
        xblocks = re.findall(r"'(XBLOCK_\d+)'", default_body)
        self.assertEqual(set(xblocks), {"XBLOCK_0", "XBLOCK_1", "XBLOCK_2"})

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        {
            "combo_kernels_autotune": 0,
            "combo_kernel_per_subkernel_blocks": True,
        }
    )
    def test_combo_kernel_no_bench_carve_out(self):
        # [FUNCTION DOC] A bucketize op must be carved out of the combo kernel (several
        # pointwise configs on CUDA, a one-element-per-thread hint on ROCm/XPU) while two
        # simple pointwise ops combine. Compiles in a fresh cache with autotuning off and
        # expects two real Triton kernels.
        # torch.bucketize triggers carve-out under all gate variants:
        #   CUDA: pointwise heuristic returns 3 configs (>2) -> carve out
        #   ROCm/XPU: lowering sets AutotuneHint.ONE_ELEMENT_PER_THREAD -> carve out
        # Simple pointwise (b*2.0, c+1.0) passes all gates -> fuses into combo.
        # Expect 1 combo + 1 standalone on all backends.
        def fn(a, b, c, boundaries):
            # [FUNCTION DOC] HELPER: Return bucketize of a against boundaries, b times 2, c
            # plus 1.
            return torch.bucketize(a, boundaries), b * 2.0, c + 1.0

        a = torch.rand(1024, device=GPU_TYPE)
        b = torch.rand(1024, device=GPU_TYPE)
        c = torch.rand(1024, device=GPU_TYPE)
        boundaries = torch.tensor([0.25, 0.5, 0.75], device=GPU_TYPE)

        out_eager = fn(a, b, c, boundaries)
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        with fresh_cache():
            out_compiled, code = run_and_get_code(
                torch.compile(fn), a, b, c, boundaries
            )
        self.assertEqual(out_eager, out_compiled)
        combined = " ".join(code)
        # 1 combo (b*2, c+1) + 1 standalone (bucketize) = 2 real kernels.
        self.assertEqual(combined.count("async_compile.triton("), 2)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
        }
    )
    def test_combo_kernel_split_large_reductions(self):
        # [FUNCTION DOC] Regression test from squeezenet backward: two very large sum
        # reductions over dims 0, 2, 3 must not be combined into one combo kernel. Builds a
        # module, compiles it with per-subkernel blocks, compares to eager (close), and expects
        # five kernels (four would mean the reductions were fused).
        # squeezenet1_1 backward (combo regression sum_sum_8bcd6e12dcd4):
        # two very large reductions (sum over [0,2,3] of [512,64,55,55]) must NOT be
        # co-fused into one combo kernel -- each is split out (5 kernels, vs 4 if fused).
        import torch._inductor.inductor_prims

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reproducing the squeezenet backward subgraph:
            # scatter-add of max pool gradients, slicing into two halves, masked selection and
            # two reductions. Has no state.
            def forward(
                self, getitem_54, arg33_1, arg61_1, full, arg62_1, sp0, sp1, sp2
            ):
                # [FUNCTION DOC] HELPER: Build a zero buffer, scatter add the viewed gradient
                # using indices recovered from low-memory max pool offsets, slice the result
                # into two channel halves, apply where masks with a scalar fill, and return the
                # sums over dims 0, 2, 3 of both halves.
                full_default = torch.ops.aten.full.default(
                    [65536, 3025], 0, dtype=torch.float32, device=getitem_54.device
                )
                view_default = torch.ops.aten.view.default(getitem_54, sp0)
                idx = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
                    arg33_1, [3, 3], [55, 55], [2, 2], [0, 0], [1, 1]
                )
                view_default_1 = torch.ops.aten.view.default(idx, sp1)
                scatter_add = torch.ops.aten.scatter_add.default(
                    full_default, 1, view_default_1, view_default
                )
                view_default_2 = torch.ops.aten.view.default(scatter_add, sp2)
                s0 = torch.ops.aten.slice.Tensor(view_default_2, 1, 0, 64)
                s1 = torch.ops.aten.slice.Tensor(view_default_2, 1, 64, 128)
                r0 = torch.ops.aten.sum.dim_IntList(
                    torch.ops.aten.where.self(arg61_1, full, s1), [0, 2, 3]
                )
                r1 = torch.ops.aten.sum.dim_IntList(
                    torch.ops.aten.where.self(arg62_1, full, s0), [0, 2, 3]
                )
                return (r0, r1)

        dev = GPU_TYPE
        torch.manual_seed(0)
        inps = (
            torch.randn(512, 128, 27, 27, device=dev),
            torch.randint(0, 9, (512, 128, 27, 27), dtype=torch.int8, device=dev),
            torch.rand(512, 64, 55, 55, device=dev) > 0.5,
            torch.randn((), device=dev),
            torch.rand(512, 64, 55, 55, device=dev) > 0.5,
            [65536, 729],
            [65536, 729],
            [512, 128, 55, 55],
        )

        m = Model()
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        out_eager = m(*inps)
        out_compiled = torch.compile(m)(*inps)
        torch.testing.assert_close(out_eager, out_compiled, rtol=1e-4, atol=1e-4)
        # Very-large reductions split out instead of co-fused: 5 kernels (4 = the regression).
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_gpu_and_triton
    @skipIfXpu(
        msg="dynamic_scale_rblock requires GPU-specific device properties "
        "(major, regs_per_multiprocessor, warp_size) not available on XPU"
    )
    @torch._inductor.config.patch(
        {
            "combo_kernel_per_subkernel_blocks": True,
        }
    )
    def test_combo_kernel_dynamic_scale_rblock(self):
        # [FUNCTION DOC] Skipped on XPU. Checks that a combo reduction kernel (per-subkernel
        # blocks, size hints absent at the autotuner level) takes part in occupancy-driven
        # reduction block scaling. Patches the autotuner constructor to collect combo
        # autotuners and the candidate iterator to run under a forced register-bound
        # single-multiprocessor device profile, then asserts a combo reduction autotuner
        # exists, can scale, produced scaled candidates and that a candidate carries a suffixed
        # per-subkernel reduction block key. Output compared to eager.
        # A combo kernel with per-subkernel blocks carries size_hints=None at the
        # autotuner level (per-subkernel hints live in combo_grid_meta), so
        # _dynamic_scale_rblock used to skip it entirely. With this change a combo
        # reduction participates in the same occupancy-driven R0_BLOCK scaling as a
        # standalone reduction.
        #
        # Two things would make a naive assertion flaky, so the candidate
        # generation is driven inside the live compile with a patched device
        # profile:
        #   1. Hardware dependence: the occupancy heuristic compares the compiled
        #      register count against regs_per_multiprocessor //
        #      max_threads_per_multi_processor, and the latter differs across GPUs
        #      (2048 on A100/H100/Blackwell vs 1536 on Ada sm_89). A forced
        #      register-bound, single-SM profile opens both hardware gates anywhere.
        #   2. Lifecycle: compile_results is cleared once the launchers are built,
        #      so _iter_rblock_scale_candidates must be exercised during the live
        #      _dynamic_scale_rblock call, while compile_results is still populated.
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Return row sums of two fused sigmoid-and-add expressions.
            r1 = (a * torch.sigmoid(a) + b).sum(dim=1)
            r2 = (c * torch.sigmoid(c) + d).sum(dim=1)
            return r1, r2

        inps = [torch.randn(8192, 2560, device=GPU_TYPE) for _ in range(4)]
        out_eager = fn(*inps)

        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        autotuners: list[CachingAutotuner] = []
        scaled_candidates = []
        orig_init = CachingAutotuner.__init__
        orig_iter = CachingAutotuner._iter_rblock_scale_candidates

        def capture_init(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement autotuner constructor: calls the original
            # and, when the metadata has combo grid meta, appends the new autotuner to the
            # collected list.
            orig_init(self, *args, **kwargs)
            if (getattr(self, "inductor_meta", {}) or {}).get("combo_grid_meta"):
                autotuners.append(self)

        def forced_iter(self):
            # [FUNCTION DOC] HELPER: Replacement candidate iterator (a generator). For a combo
            # reduction without size hints it temporarily swaps in a device profile with one
            # multiprocessor and a huge thread limit, collects candidates from the original
            # iterator into the shared list, restores the profile (even on error) and yields
            # them; otherwise it delegates to the original.
            # For the combo reduction (size_hints=None), force a register-bound,
            # single-SM profile so candidate generation is deterministic across
            # GPUs, then restore the real profile before the caller precompiles.
            if self.size_hints is None and self._combo_has_reduction_subkernel:
                orig_props = self.device_props
                self.device_props = orig_props._replace(
                    multi_processor_count=1,
                    max_threads_per_multi_processor=1 << 30,
                )
                try:
                    candidates = list(orig_iter(self))
                finally:
                    self.device_props = orig_props
                scaled_candidates.extend(candidates)
                yield from candidates
            else:
                yield from orig_iter(self)

        with (
            fresh_cache(),  # isolate from cache so a re-run (inherited subclass) recompiles
            patch.object(CachingAutotuner, "__init__", capture_init),
            patch.object(
                CachingAutotuner, "_iter_rblock_scale_candidates", forced_iter
            ),
        ):
            torch._dynamo.reset()
            out_compiled = torch.compile(fn)(*inps)

        torch.testing.assert_close(out_eager, out_compiled, atol=1e-3, rtol=1e-3)

        combo_reductions = [
            au for au in autotuners if au._combo_has_reduction_subkernel
        ]
        self.assertTrue(
            combo_reductions,
            "expected a combo reduction kernel with per-subkernel blocks",
        )
        self.assertTrue(
            any(au._could_rblock_scale for au in combo_reductions),
            "_could_rblock_scale should be True for a combo reduction",
        )
        self.assertTrue(
            scaled_candidates,
            "_dynamic_scale_rblock should generate a scaled R0_BLOCK candidate "
            "for the combo reduction",
        )
        # Combo reductions emit per-subkernel suffixed blocks (R0_BLOCK_i), not the
        # bare R0_BLOCK of a standalone reduction.
        self.assertTrue(
            any(
                any(k.startswith("R0_BLOCK_") for k in cfg.kwargs)
                for cfg in scaled_candidates
            ),
            "combo scaled candidate should carry a suffixed R0_BLOCK_i kwarg",
        )


class ComboKernelBenchmarkTests(TestCase):
    # [CLASS DOC] Same combo kernel scenarios as the main tests but with combo kernel
    # benchmarking turned on (config benchmark_combo_kernel), which generates extra throwaway
    # kernels and so changes expected kernel counts. Per-subkernel blocks default off via a
    # class attribute.
    check_model_gpu = check_model_gpu
    check_model_cpu = check_model
    check_kernel_count = True
    combo_kernel_per_subkernel_blocks = False

    def setUp(self):
        # [FUNCTION DOC] Reset dynamo and Inductor metrics, then push a config patch enabling
        # combo kernels with benchmarking, unlimited distance and no peak-memory limits onto
        # an ExitStack.
        super().setUp()
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                    "combo_kernel_per_subkernel_blocks": self.combo_kernel_per_subkernel_blocks,
                    "combo_kernel_max_distance": -1,
                    "combo_kernel_peak_memory_increase_gb": None,
                    "combo_kernel_peak_memory_pct_threshold": None,
                }
            )
        )

    def tearDown(self):
        # [FUNCTION DOC] Close the config patch stack, reset dynamo and metrics, call
        # base tearDown.
        self._test_stack.close()
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    def test_activation_benchmark(self):
        # [FUNCTION DOC] Relu, sigmoid, tanh on three inputs matches eager and generates five
        # kernels (benchmarking inflates the count).
        def test_activations(a, b, c):
            # [FUNCTION DOC] HELPER: Apply relu, sigmoid and tanh to three inputs and return
            # the results.
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_gpu_and_triton
    def test_reduce_benchmark(self):
        # [FUNCTION DOC] Sum, max, min reductions plus tanh match eager; kernel count must be
        # between five and ten inclusive.
        def test_reduce(a, b, c, d):
            # [FUNCTION DOC] HELPER: Return sum, max (values and indices), min and tanh of the
            # four inputs.
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10)

    @requires_gpu_and_triton
    def test_mutated_benchmark(self):
        # [FUNCTION DOC] Mutating ops on four inputs match eager; kernel count between five
        # and ten.
        def test_mutated(a, b, c, d):
            # [FUNCTION DOC] HELPER: Mutate three inputs in place, add 5 to the third, return
            # all four.
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10)

    @requires_gpu_and_triton
    def test_round_robin_dispatch(self):
        # [FUNCTION DOC] Exercises the round-robin dispatch strategy of combo kernels with
        # mutated inputs of varied shapes: output equals eager and six kernels are generated.
        # combo kernel dispatch strategy: round robin
        def test_mutated(a, b, c, d):
            # [FUNCTION DOC] HELPER: Mutate three inputs in place, add 5 to the third, return
            # all four.
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 5, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(5, 18, device=GPU_TYPE),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_gpu_and_triton
    def test_2d_blocking_benchmark(self):
        # [FUNCTION DOC] Three adds with two transposed operands through the model checker; six
        # kernels generated with benchmarking enabled.
        def fn(a0, a1, a2, b0, b1, b2):
            # [FUNCTION DOC] HELPER: Add three pairs of tensors and return the sums.
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_gpu(
            fn,
            (
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(40, 30, device=GPU_TYPE),
                torch.rand(36, 40, device=GPU_TYPE),
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(30, 40, device=GPU_TYPE).t(),
                torch.rand(40, 36, device=GPU_TYPE).t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_gpu_and_triton
    def test_persistent_reduction_no_x_dim(self):
        # [FUNCTION DOC] Row sums of two inputs with a dynamic leading dimension (marked
        # dynamic with bounds). Persistent reductions without an x dimension still match eager;
        # four kernels.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the dim 1 sums of both inputs.
            return x.sum(1), y.sum(1)

        inps = (
            torch.rand(16, 256, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @requires_gpu_and_triton
    def test_graph_state_restored_on_swallowed_compilation_error(self):
        # [FUNCTION DOC] Verifies the scheduler restores the graph removed-buffer state when
        # the benchmark codegen fails with a swallowed loop-carried-variable compilation error.
        # Patches speedup_by_combo_kernel to record whether the removed buffers object is the
        # same afterwards, and patches combo kernel code generation so only-source codegen
        # mutates the removed buffers and raises a fake error (real codegen passes through).
        # Asserts the state was restored.
        # speedup_by_combo_kernel swallows Loop-carried-variable CompilationError
        # and keeps compiling; benchmark_combo_kernel must restore the original
        # V.graph.removed_buffers/inplaced_to_remove on that path, not leak the
        # throwaway copies it benchmarks with.
        from triton.compiler.errors import CompilationError

        from torch._inductor.codegen.simd import SIMDScheduling
        from torch._inductor.scheduler import Scheduler

        class FakeCompilationError(CompilationError):
            # [CLASS DOC] HELPER: Fake triton CompilationError that skips the base constructor
            # and keeps only a message, so it can be raised without a real compiler context.
            def __init__(self, msg):
                # [FUNCTION DOC] HELPER: Initialize only the base Exception with the message
                # and store it.
                Exception.__init__(self, msg)
                self.msg = msg

            def __str__(self):
                # [FUNCTION DOC] HELPER: Return the stored message.
                return self.msg

        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Return a times 2, b plus 1, c minus 1.
            return a * 2.0, b + 1.0, c - 1.0

        inps = [torch.rand(64, device=GPU_TYPE) for _ in range(3)]

        captured = {}
        orig_speedup = Scheduler.speedup_by_combo_kernel
        orig_generate = SIMDScheduling.generate_combo_kernel_code

        def speedup_capturing_restore(sched, nodes):
            # [FUNCTION DOC] HELPER: Wrapper around the original scheduler speedup check.
            # Records the result and whether the graph removed-buffers object is the same
            # object after the call; returns the result.
            before = V.graph.removed_buffers
            result = orig_speedup(sched, nodes)
            captured["result"] = result
            captured["restored"] = V.graph.removed_buffers is before
            return result

        def mutate_then_raise(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement combo codegen. Delegates to the original
            # unless the call asks for source only; in that case adds a throwaway name to the
            # graph removed buffers and raises the fake compilation error mentioning a
            # loop-carried variable.
            # Real codegen (only_gen_src_code=False) must keep working; only
            # the benchmark's throwaway codegen simulates mutate-then-fail.
            if not kwargs.get("only_gen_src_code"):
                return orig_generate(self, *args, **kwargs)
            V.graph.removed_buffers.add("throwaway_buf")
            raise FakeCompilationError("Loop-carried variable")

        with (
            patch.object(
                Scheduler, "speedup_by_combo_kernel", speedup_capturing_restore
            ),
            patch.object(
                SIMDScheduling, "generate_combo_kernel_code", mutate_then_raise
            ),
            fresh_cache(),
        ):
            out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(fn(*inps), out_compiled)
        self.assertTrue(captured["result"])
        self.assertTrue(captured["restored"])


class ComboKernelDynamicShapesTests(TestCase):
    # [CLASS DOC] Combo kernel tests with dynamic shapes: benchmarking is on, automatic dynamic
    # shapes are off and static-by-default is off, so all sizes are symbolic. Expected kernel
    # counts include benchmark kernels. Per-subkernel blocks come from a class attribute
    # (default off).
    check_model_gpu = check_model_gpu
    check_model_cpu = check_model
    check_kernel_count = True
    combo_kernel_per_subkernel_blocks = False

    def setUp(self):
        # [FUNCTION DOC] Reset Inductor metrics, then push onto an ExitStack an Inductor config
        # patch (combo kernels on, benchmarking on, unlimited distance, no peak-memory limits)
        # and a dynamo config patch (no automatic dynamic shapes, not static by default).
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                    "combo_kernel_per_subkernel_blocks": self.combo_kernel_per_subkernel_blocks,
                    "combo_kernel_max_distance": -1,
                    "combo_kernel_peak_memory_increase_gb": None,
                    "combo_kernel_peak_memory_pct_threshold": None,
                }
            )
        )
        self._test_stack.enter_context(
            torch._dynamo.config.patch(
                {
                    "automatic_dynamic_shapes": False,
                    "assume_static_by_default": False,
                }
            )
        )

    def tearDown(self):
        # [FUNCTION DOC] Close the config patch stack, reset Inductor metrics, call
        # base tearDown.
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    def test_dynamic_shapes_activations(self):
        # [FUNCTION DOC] Relu, sigmoid, tanh under dynamic shapes match eager; five
        # kernels generated.
        def test_activations(a, b, c):
            # [FUNCTION DOC] HELPER: Apply relu, sigmoid and tanh to three inputs and return
            # the results.
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_gpu_and_triton
    def test_dynamic_shapes_2d_blocking(self):
        # [FUNCTION DOC] Three adds with two transposed operands under dynamic shapes, via the
        # model checker; six kernels generated.
        def fn(a0, a1, a2, b0, b1, b2):
            # [FUNCTION DOC] HELPER: Add three pairs of tensors and return the sums.
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_gpu(
            fn,
            (
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(40, 30, device=GPU_TYPE),
                torch.rand(36, 40, device=GPU_TYPE),
                torch.rand(30, 20, device=GPU_TYPE),
                torch.rand(30, 40, device=GPU_TYPE).t(),
                torch.rand(40, 36, device=GPU_TYPE).t(),
            ),
        )
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_gpu_and_triton
    def test_dynamic_shapes_reduce(self):
        # [FUNCTION DOC] Sum, max, min and tanh under dynamic shapes match eager; between five
        # and ten kernels.
        def test_reduce(a, b, c, d):
            # [FUNCTION DOC] HELPER: Return sum, max, min over dim 0 and tanh of the
            # fourth input.
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(30, 8, device=GPU_TYPE),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10)

    @skipIfXpu(msg="https://github.com/pytorch/pytorch/issues/181863")
    @requires_gpu_and_triton
    def test_dynamic_shapes_mutated(self):
        # [FUNCTION DOC] Skipped on XPU (known issue link). Mutating ops with round-robin
        # dispatch under dynamic shapes: output equals eager, six kernels generated.
        # combo kernel dispatch strategy: round robin
        def test_mutated(a, b, c, d):
            # [FUNCTION DOC] HELPER: Mutate three inputs in place, add 5 to the third, return
            # all four.
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 5, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(5, 18, device=GPU_TYPE),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_gpu_and_triton
    @torch._inductor.config.patch("combo_kernels_autotune", 0)
    def test_dynamic_shapes_activations_no_autotune(self):
        # [FUNCTION DOC] With combo autotuning off, three activations compile to one Triton
        # kernel (counted via async compile calls in the generated code) and match eager.
        def test_activations(a, b, c):
            # [FUNCTION DOC] HELPER: Apply relu, sigmoid and tanh to three inputs and return
            # the results.
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = test_activations(*inps)
        out_compiled, code = run_and_get_code(torch.compile(test_activations), *inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(sum(s.count("async_compile.triton(") for s in code), 1)

    @requires_gpu_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_persistent_reduction_no_x_dim(self):
        # [FUNCTION DOC] Automatic dynamic shapes on, static by default on; two row sums with
        # the first dimension marked dynamic. Persistent reduction lacking an x dimension
        # matches eager; four kernels.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the dim 1 sums of both inputs.
            return x.sum(1), y.sum(1)

        inps = (
            torch.rand(16, 256, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @requires_gpu_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_persistent_reduction_no_x_dim_2(self):
        # [FUNCTION DOC] Like the previous test with 3D inputs and dims 0 and 1 marked dynamic,
        # reducing the last dim; output matches eager and four kernels are generated.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the dim 2 sums of both inputs.
            return x.sum(2), y.sum(2)

        inps = (
            torch.rand(8, 16, 256, device=GPU_TYPE),
            torch.rand(8, 32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], (0, 1), min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], (0, 1), min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @requires_gpu_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_2d_blocking_round_robin(self):
        # [FUNCTION DOC] Three adds with transposed operands, compiled once and then called
        # with a second set of different shapes (recompiles dynamically). Both runs match eager
        # and the kernel count (reset between runs) is five or six each time.
        def fn(a0, a1, a2, b0, b1, b2):
            # [FUNCTION DOC] HELPER: Add three pairs of tensors and return the sums.
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        inps = (
            torch.rand(20, 30, device=GPU_TYPE),
            torch.rand(30, 30, device=GPU_TYPE),
            torch.rand(40, 32, device=GPU_TYPE),
            torch.rand(30, 20, device=GPU_TYPE).t(),
            torch.rand(30, 30, device=GPU_TYPE).t(),
            torch.rand(32, 40, device=GPU_TYPE).t(),
        )

        out_eager = fn(*inps)
        compiled = torch.compile(fn)
        out_compiled = compiled(*inps)
        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(5 <= torch._inductor.metrics.generated_kernel_count <= 6)
        torch._inductor.metrics.reset()

        inps = (
            torch.rand(24, 30, device=GPU_TYPE),
            torch.rand(32, 30, device=GPU_TYPE),
            torch.rand(48, 32, device=GPU_TYPE),
            torch.rand(30, 24, device=GPU_TYPE).t(),
            torch.rand(30, 32, device=GPU_TYPE).t(),
            torch.rand(32, 48, device=GPU_TYPE).t(),
        )
        out_compiled = compiled(*inps)
        out_eager = fn(*inps)
        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(5 <= torch._inductor.metrics.generated_kernel_count <= 6)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    @torch._inductor.config.patch("triton.autotune_at_compile_time", True)
    def test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda(self):
        # [FUNCTION DOC] CUDA only. Row sum, mean and max over different widths with the
        # first dim marked dynamic and compile-time autotuning enabled; only checks output
        # equals eager.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Return sum, mean and max over dim 1 of the three inputs.
            return x.sum(1), y.mean(1), z.max(1)

        inps = (
            torch.rand(16, 128, device=GPU_TYPE),
            torch.rand(32, 128, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[2], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

    @requires_xpu_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    @torch._inductor.config.patch("triton.autotune_at_compile_time", True)
    def test_dynamic_shapes_persistent_reduction_mixed_x_dim_xpu(self):
        # [FUNCTION DOC] XPU variant of the previous test: same function and checks,
        # XPU-only decorator.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Return sum, mean and max over dim 1 of the three inputs.
            return x.sum(1), y.mean(1), z.max(1)

        inps = (
            torch.rand(16, 128, device=GPU_TYPE),
            torch.rand(32, 128, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[2], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

    @requires_gpu_and_triton
    def test_helper_fn_defined(self):
        # [FUNCTION DOC] A sum, a mean and a cumulative sum in one function: output matches
        # eager and the generated code defines the scan add helper function exactly once
        # (no duplicates).
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Return sum, mean and cumulative sum over dim 1 of the
            # three inputs.
            return x.sum(1), y.mean(1), z.cumsum(1)

        inps = (
            torch.rand(16, 128, device=GPU_TYPE),
            torch.rand(32, 128, device=GPU_TYPE),
            torch.rand(32, 256, device=GPU_TYPE),
        )

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        code = " ".join(code)
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(code.count("def _triton_helper_fn_add0(arg0_0, arg1_0):"), 1)

    @requires_gpu_and_triton
    def test_dynamic_shapes_persistent_reduction_dynamic_rdim(self):
        # [FUNCTION DOC] Loops over benchmarking off and on, resetting dynamo and metrics. With
        # dynamic compilation and shape upper bounds of 32, two full sums must be correct,
        # produce 1 kernel (4 with benchmarking) and the code must declare the reduction block
        # constant with value 32.
        # KNOWN ISSUE: the code check branches on the per-subkernel-blocks config in the
        # opposite sense to the other tests: the unsuffixed block name is expected when
        # the option is on and the suffixed names when it is off. Not verified whether
        # this is intended.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Assert both leading sizes are at most 32 and return the
            # full sums.
            torch._check(a.shape[0] <= 32)
            torch._check(b.shape[0] <= 32)
            return a.sum(), b.sum()

        for benchmark in (False, True):
            torch._dynamo.reset()
            torch._inductor.metrics.reset()
            a = torch.randn(16, device=GPU_TYPE)
            b = torch.randn(16, device=GPU_TYPE)
            with torch._inductor.config.patch("benchmark_combo_kernel", benchmark):
                fn_c = torch.compile(fn, dynamic=True)
                result, code = run_and_get_code(fn_c, a, b)
                self.assertEqual(result[0], a.sum())
                self.assertEqual(result[1], b.sum())
                self.assertEqual(
                    torch._inductor.metrics.generated_kernel_count,
                    4 if benchmark else 1,
                )
                if torch._inductor.config.combo_kernel_per_subkernel_blocks:
                    FileCheck().check_count(
                        "R0_BLOCK: tl.constexpr = 32", 1, exactly=True
                    ).run(code[0])
                else:
                    FileCheck().check("R0_BLOCK_0: tl.constexpr = 32").check(
                        "R0_BLOCK_1: tl.constexpr = 32"
                    ).run(code[0])


class ComboKernelTestsPerSubkernelBlocks(ComboKernelTests):
    # [CLASS DOC] Re-runs all main combo kernel tests with per-subkernel blocks enabled, and
    # adds tests that check each sub-kernel body is emitted as a separate non-inlined Triton
    # device function called from its dispatch branch (so register allocation is per body).
    combo_kernel_per_subkernel_blocks = True

    @requires_gpu_and_triton
    def test_noinline_sub_kernel_bodies(self):
        # [FUNCTION DOC] Three pointwise ops of different sizes: asserts via FileCheck that
        # three no-inline function definitions come first, then the program id read, then
        # dispatch branches each calling its body. Output must match eager.
        # Per-subkernel combos emit each body as a @triton.jit(noinline=True)
        # device function called from its dispatch branch, so ptxas allocates
        # registers per body instead of jointly over the merged control-flow
        # graph (a register-fat member cannot degrade the others' occupancy).
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Return x plus 1, sigmoid of y, tanh of z.
            return x + 1, torch.sigmoid(y), torch.tanh(z)

        inps = (
            torch.rand(8192, device=GPU_TYPE),
            torch.rand(6144, device=GPU_TYPE),
            torch.rand(4096, device=GPU_TYPE),
        )

        out, code = run_and_get_code(torch.compile(fn), *inps)
        self.assertEqual(out, fn(*inps))
        code = " ".join(code)

        # Three sub-function definitions before the combo kernel, each
        # dispatch branch reduced to a call.
        fc = FileCheck()
        for num in range(3):
            fc = fc.check("@triton.jit(noinline=True)").check(f"_body_{num}(")
        fc = fc.check("pid = tl.program_id(0)")
        for num in range(3):
            fc = fc.check(f"pid < num_blocks_{num}:").check(f"_body_{num}(")
        fc.run(code)

    @requires_gpu_and_triton
    def test_noinline_sub_kernel_body_dedupes_identical_members(self):
        # [FUNCTION DOC] Two identical add-one members: only one no-inline body function is
        # emitted, and both dispatch branches call it (deduplication of identical bodies).
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return x plus 1 and y plus 1.
            return x + 1, y + 1

        inps = (
            torch.rand(8192, device=GPU_TYPE),
            torch.rand(8192, device=GPU_TYPE),
        )

        out, code = run_and_get_code(torch.compile(fn), *inps)
        self.assertEqual(out, fn(*inps))
        code = " ".join(code)

        (
            FileCheck()
            .check_count("@triton.jit(noinline=True)", 1, exactly=True)
            .check("_body_0(")
            .check("pid = tl.program_id(0)")
            .check("pid < num_blocks_0:")
            .check("_body_0(")
            .check("pid < num_blocks_1:")
            .check("_body_0(")
            .run(code)
        )

    @requires_gpu_and_triton
    def test_noinline_sub_kernel_body_dedupes_by_signature(self):
        # [FUNCTION DOC] Two add-one members with float32 vs float64 inputs: bodies differ by
        # signature so two no-inline functions are emitted, one per dispatch branch.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return x plus 1 and y plus 1.
            return x + 1, y + 1

        inps = (
            torch.rand(8192, device=GPU_TYPE, dtype=torch.float32),
            torch.rand(8192, device=GPU_TYPE, dtype=torch.float64),
        )

        out, code = run_and_get_code(torch.compile(fn), *inps)
        self.assertEqual(out, fn(*inps))
        code = " ".join(code)

        FileCheck().check_count("@triton.jit(noinline=True)", 2, exactly=True).check(
            "_body_0("
        ).check("_body_1(").run(code)
        (
            FileCheck()
            .check("pid = tl.program_id(0)")
            .check("pid < num_blocks_0:")
            .check("_body_0(")
            .check("pid < num_blocks_1:")
            .check("_body_1(")
            .run(code)
        )

    @requires_gpu_and_triton
    @torch._inductor.config.patch("combo_kernels_autotune", 0)
    def test_noinline_sub_kernel_bodies_no_bench(self):
        # [FUNCTION DOC] With combo autotuning off, two members of different sizes: checks body
        # function definitions and signature block parameters (shared and per-subkernel, with
        # no baked default for the first), then dispatch branches calling the matching body.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return x plus 1 and tanh of y.
            return x + 1, torch.tanh(y)

        inps = (
            torch.rand(8192, device=GPU_TYPE),
            torch.rand(4096, device=GPU_TYPE),
        )

        out, code = run_and_get_code(torch.compile(fn), *inps)
        self.assertEqual(out, fn(*inps))
        code = " ".join(code)

        (
            FileCheck()
            .check("@triton.jit(noinline=True)")
            .check("_body_0(")
            .check("XBLOCK : tl.constexpr")
            .check("@triton.jit(noinline=True)")
            .check("_body_1(")
            .check("XBLOCK : tl.constexpr")
            .check("XBLOCK_1 : tl.constexpr")
            .check_not("XBLOCK_0: tl.constexpr =")
            .check("pid < num_blocks_0:")
            .check("_body_0(")
            .check("pid < num_blocks_1:")
            .check("_body_1(")
            .run(code)
        )


class ComboKernelBenchmarkTestsPerSubkernelBlocks(ComboKernelBenchmarkTests):
    # [CLASS DOC] Re-runs the benchmark combo kernel tests with per-subkernel blocks enabled
    # and adds a test on source-only codegen configs for regular (non-persistent) reductions.
    combo_kernel_per_subkernel_blocks = True

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        {
            "combo_kernels_autotune": 0,
            "max_autotune": False,
            "max_autotune_pointwise": False,
            "triton.persistent_reductions": False,
        }
    )
    def test_no_autotune_regular_reduction(self):
        # [FUNCTION DOC] With autotuning, max_autotune and persistent reductions off, two row
        # sums of 2048 columns. Wraps combo kernel code generation to record, for source-only
        # calls, the change in generated kernel count and the stitched config kwargs of
        # multi-node kernels. Asserts the output close to eager, exactly one such config, a
        # kernel count delta of one, and that its reduction block keys are exactly the two
        # per-subkernel ones.
        from torch._inductor.codegen.simd import SIMDScheduling

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return last-dim sums of both inputs.
            return a.sum(-1), b.sum(-1)

        source_only_configs = []
        source_only_kernel_deltas = []
        orig_generate = SIMDScheduling.generate_combo_kernel_code

        def capture_source_only_config(scheduling, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper over the original combo codegen; for source-only
            # calls records the kernel count delta and the stitched-config kwargs of results
            # with several nodes, and returns the original result unchanged.
            old_kernel_count = torch._inductor.metrics.generated_kernel_count
            result = orig_generate(scheduling, *args, **kwargs)
            if kwargs.get("only_gen_src_code"):
                source_only_kernel_deltas.append(
                    torch._inductor.metrics.generated_kernel_count - old_kernel_count
                )
                source_only_configs.extend(
                    kernel.no_bench_stitched_config.kwargs
                    for _, kernel, nodes in result
                    if kernel is not None
                    and len(nodes) > 1
                    and kernel.no_bench_stitched_config is not None
                )
            return result

        inps = (
            torch.randn(512, 2048, device=GPU_TYPE),
            torch.randn(256, 2048, device=GPU_TYPE),
        )
        expected = fn(*inps)

        with (
            fresh_cache(),
            patch.object(
                SIMDScheduling,
                "generate_combo_kernel_code",
                capture_source_only_config,
            ),
        ):
            actual = torch.compile(fn)(*inps)

        self.assertEqual(actual, expected, atol=2e-5, rtol=2e-5)
        self.assertEqual(len(source_only_configs), 1)
        self.assertEqual(source_only_kernel_deltas, [1])
        self.assertEqual(
            {key for key in source_only_configs[0] if key.startswith("R0_BLOCK")},
            {"R0_BLOCK_0", "R0_BLOCK_1"},
        )


class ComboKernelDynamicShapesTestsPerSubkernelBlocks(ComboKernelDynamicShapesTests):
    # [CLASS DOC] Re-runs the dynamic shape combo tests with per-subkernel blocks enabled.
    combo_kernel_per_subkernel_blocks = True


@instantiate_parametrized_tests
class ComboKernelCompileTimeAutotuneTests(TestCase):
    # [CLASS DOC] Tests for compile-time per-subkernel autotuning of combo kernels: each
    # subkernel is tuned standalone and winners are stitched into the combo launch config.
    # setUp enables combo kernels, per-subkernel blocks and compile-time autotune with
    # benchmarking off.
    """Compile-time per-subkernel autotune: each combo subkernel autotunes its blocks
    standalone at compile time; winners are stitched into the combo's launch config."""

    def setUp(self):
        # [FUNCTION DOC] Reset metrics and push a config patch (combo kernels, no benchmarking,
        # per-subkernel blocks and compile-time autotune on) onto an ExitStack.
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                    "combo_kernel_per_subkernel_blocks": True,
                    "combo_kernel_compile_time_autotune": True,
                }
            )
        )

    def tearDown(self):
        # [FUNCTION DOC] Close the patch stack, reset metrics, call base tearDown.
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    @parametrize("mode", ["default", "cdt"])
    def test_compile_time_autotune(self, mode):
        # [FUNCTION DOC] Parametrized by mode: default, or coordinate descent tuning enabled.
        # In a fresh cache compiles a function that gives one pointwise combo and one reduction
        # combo (four subkernels with differing shapes). Asserts output equals eager, that the
        # compile-time autotune counters (tuned plus cached) total four with zero fallbacks,
        # and, for the coordinate descent mode, that the coordinate descent benchmark counter
        # is positive.
        extra = {}
        if mode == "cdt":
            extra["coordinate_descent_tuning"] = True

        # one fn -> a pointwise combo (a+b, c*d) AND a reduction combo (e.sum, g.amax),
        # each with differently-shaped subkernels so their blocks differ.
        def f(a, b, c, d, e, g):
            # [FUNCTION DOC] HELPER: Return a plus b, c times d, last-dim sum of e and last-dim
            # max of g.
            return a + b, c * d, e.sum(-1), g.amax(-1)

        inps = [
            torch.randn(8192, device=GPU_TYPE),
            torch.randn(8192, device=GPU_TYPE),
            torch.randn(4096, device=GPU_TYPE),
            torch.randn(4096, device=GPU_TYPE),
            torch.randn(1024, 512, device=GPU_TYPE),
            torch.randn(1024, 768, device=GPU_TYPE),
        ]
        counters.clear()
        # fresh_cache so the subkernels aren't served from the autotune cache
        # (.best_config), which is what makes the cdt mode actually run coordesc.
        with fresh_cache(), torch._inductor.config.patch(extra):
            out, code = run_and_get_code(torch.compile(f), *inps)
        code = " ".join(code)

        self.assertEqual(out, f(*inps))
        # all 4 subkernels were processed by the compile-time autotuner
        processed = (
            counters["inductor"]["combo_subkernel_autotune"]
            + counters["inductor"]["combo_subkernel_autotune_cached"]
        )
        self.assertEqual(processed, 4)
        self.assertEqual(counters["inductor"]["combo_subkernel_autotune_fallback"], 0)
        if mode == "cdt":
            self.assertGreater(counters["inductor"]["coordesc_tuning_bench"], 0)

    @requires_gpu_and_triton
    def test_compile_time_autotune_caching(self):
        # [FUNCTION DOC] Checks the on-disk autotune cache for compile-time subkernel tuning.
        # With max autotune on and graph caches off in a fresh cache, a cold compile must tune
        # at least one subkernel with no fallbacks. After resetting dynamo and purging only the
        # in-memory code cache, a warm compile must tune none and report the same number as
        # cached, again with no fallbacks.
        from torch._inductor.codecache import PyCodeCache

        def f(a, b, c, d):
            # [FUNCTION DOC] HELPER: Return a plus b and c times d.
            return a + b, c * d

        # shapes distinct from the other test so these subkernels aren't already cached
        inps = [
            torch.randn(2048, device=GPU_TYPE),
            torch.randn(2048, device=GPU_TYPE),
            torch.randn(3072, device=GPU_TYPE),
            torch.randn(3072, device=GPU_TYPE),
        ]
        # disable FXGraphCache so codegen re-runs and we exercise the autotune cache
        with (
            fresh_cache(),
            torch._inductor.config.patch(
                {
                    "max_autotune": True,
                    "fx_graph_cache": False,
                    "fx_graph_remote_cache": False,
                }
            ),
        ):
            counters.clear()
            torch.compile(f)(*inps)  # cold: subkernels benchmarked
            cold = counters["inductor"]["combo_subkernel_autotune"]
            self.assertGreater(cold, 0)
            self.assertEqual(
                counters["inductor"]["combo_subkernel_autotune_fallback"], 0
            )

            # Simulate a new process: drop the in-memory module cache (which would
            # otherwise serve the already-tuned autotuner object) but keep the on-disk
            # caches, so the warm run genuinely re-imports and reads .best_config.
            torch._dynamo.reset()
            PyCodeCache.cache_clear(purge=False)
            counters.clear()
            torch.compile(f)(*inps)  # warm: reuse .best_config, no re-benchmark
            self.assertEqual(counters["inductor"]["combo_subkernel_autotune"], 0)
            warm_cached = counters["inductor"]["combo_subkernel_autotune_cached"]
            self.assertEqual(warm_cached, cold)
            self.assertEqual(
                counters["inductor"]["combo_subkernel_autotune_fallback"], 0
            )

    @requires_gpu_and_triton
    def test_compile_time_autotune_looped_reduction(self):
        # [FUNCTION DOC] Row sums/max of 8192 and 16384 wide rows (looped, not persistent, not
        # split) compile in a fresh cache; result close to eager (loose tolerance). Unless
        # cpp_wrapper is on, the generated source must mention both per-subkernel reduction
        # block arguments in order. No autotune fallback may be counted.
        # rnumel above the persistent threshold but below the split-reduction
        # threshold -> single-pass looped reductions, whose winning configs
        # carry R0_BLOCK; the combo must expose R0_BLOCK_i as constexpr args.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return the last-dim sum of a and last-dim max of b.
            return a.sum(-1), b.amax(-1)

        inps = [
            torch.randn(64, 8192, device=GPU_TYPE),
            torch.randn(64, 16384, device=GPU_TYPE),
        ]
        counters.clear()
        with fresh_cache():
            out, code = run_and_get_code(torch.compile(f), *inps)
        # loose tolerance: summing 65536 fp32 elements is accumulation-order sensitive
        self.assertEqual(out, f(*inps), atol=1e-3, rtol=1e-3)
        # cpp_wrapper emits C++ source; the constexpr args only appear in python codegen
        if not torch._inductor.config.cpp_wrapper:
            src = " ".join(code)
            FileCheck().check("R0_BLOCK_0").check("R0_BLOCK_1").run(src)
        self.assertEqual(counters["inductor"]["combo_subkernel_autotune_fallback"], 0)

    @requires_gpu_and_triton
    def test_compile_time_autotune_dynamic_shapes(self):
        # [FUNCTION DOC] Two pointwise ops with dynamic sizes compile once in a fresh cache
        # with dynamic mode; the result matches eager, calling again with new sizes also
        # matches, and no fallback is counted.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return a plus 1 and b times 2.
            return a + 1.0, b * 2.0

        a = torch.randn(4096, device=GPU_TYPE)
        b = torch.randn(6144, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        counters.clear()
        with fresh_cache():
            fn_c = torch.compile(f, dynamic=True)
            out = fn_c(a, b)
        self.assertEqual(out, f(a, b))
        # recompile-free on new dynamic sizes
        a2 = torch.randn(5000, device=GPU_TYPE)
        b2 = torch.randn(7000, device=GPU_TYPE)
        self.assertEqual(fn_c(a2, b2), f(a2, b2))
        self.assertEqual(counters["inductor"]["combo_subkernel_autotune_fallback"], 0)

    @requires_gpu_and_triton
    def test_compile_time_autotune_sort_forces_persistent(self):
        # [FUNCTION DOC] Sort members must stay persistent in the compile-time benchmark twin
        # even if the choices heuristic says non-persistent. Patches the heuristic to return
        # false for kernels containing sort, disables coalesced tiling analysis, compiles two
        # sorts; expects eager-equal output and zero autotune fallbacks (sorts stay fused in
        # the combo).
        # The standalone benchmark twin must stay persistent for sort even when the
        # choices-level heuristic says otherwise (TritonKernel.sort asserts persistent;
        # TritonKernel.should_use_persistent_reduction enforces it above the heuristic).
        from torch._inductor.choices import InductorChoices

        orig = InductorChoices.should_use_persistent_reduction

        @staticmethod
        def force_non_persistent_for_sort(features, cooperative_reduction):
            # [FUNCTION DOC] HELPER: Replacement persistent-reduction heuristic: false when the
            # features contain sort, otherwise the saved original heuristic result.
            if features.contains_op("sort"):
                return False
            return orig(features, cooperative_reduction)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return sorted values along dim 1 of both inputs.
            return torch.sort(a, dim=1).values, torch.sort(b, dim=1).values

        inps = [
            torch.randint(0, 1000, (10, 200), device=GPU_TYPE),
            torch.randint(0, 1000, (20, 200), device=GPU_TYPE),
        ]
        counters.clear()
        with (
            patch.object(
                InductorChoices,
                "should_use_persistent_reduction",
                force_non_persistent_for_sort,
            ),
            fresh_cache(),
            torch._inductor.config.patch({"triton.coalesce_tiling_analysis": False}),
        ):
            out = torch.compile(fn)(*inps)
        self.assertEqual(out, fn(*inps))
        # Sorts stay fused in the combo: tuned normally, no carve-out fallbacks.
        self.assertEqual(counters["inductor"]["combo_subkernel_autotune_fallback"], 0)

    @requires_gpu_and_triton
    @parametrize("benchmark_combo_kernel", [False, True])
    def test_benchmarking_excludes_indirect_indexing(self, benchmark_combo_kernel):
        # [FUNCTION DOC] Parametrized by benchmark_combo_kernel (compile-time autotune is set
        # to the opposite). A gather with an indirect index plus two pointwise ops: result
        # matches eager, and the generated code shows the gather as its own kernel while the
        # combo has only 2 subkernels, so indirect indexing is excluded from benchmarking in
        # both modes.
        def fn(a, b, c, idx):
            # [FUNCTION DOC] HELPER: Return a gathered at idx plus 1024, b times 2, c plus 1.
            return a[idx + 1024], b * 2.0, c + 1.0

        inps = [
            torch.randn(1024, device=GPU_TYPE),
            torch.randn(256, device=GPU_TYPE),
            torch.randn(256, device=GPU_TYPE),
            # Real indices map to zero; synthetic zero-filled indices map out of bounds.
            torch.full((256,), -1024, device=GPU_TYPE, dtype=torch.int64),
        ]
        with (
            fresh_cache(),
            torch._inductor.config.patch(
                {
                    "benchmark_combo_kernel": benchmark_combo_kernel,
                    "combo_kernel_compile_time_autotune": not benchmark_combo_kernel,
                }
            ),
        ):
            out, code = run_and_get_code(torch.compile(fn), *inps)
        self.assertEqual(out, fn(*inps))
        src = code[0]
        # The gather is its own kernel; the combo stitches only the 2 pointwise subkernels.
        FileCheck().check("triton_poi_fused_add_index").check("'num_kernels': 2").run(
            src
        )

    @requires_gpu_and_triton
    def test_disabled_autotune_keeps_indirect_indexing(self):
        # [FUNCTION DOC] With combo autotuning disabled, the same gather plus two pointwise ops
        # stay in one combo kernel of 3 subkernels, with output equal to eager.
        def fn(a, b, c, idx):
            # [FUNCTION DOC] HELPER: Return a gathered at idx plus 1024, b times 2, c plus 1.
            return a[idx + 1024], b * 2.0, c + 1.0

        inps = [
            torch.randn(1024, device=GPU_TYPE),
            torch.randn(256, device=GPU_TYPE),
            torch.randn(256, device=GPU_TYPE),
            torch.full((256,), -1024, device=GPU_TYPE, dtype=torch.int64),
        ]
        with fresh_cache(), torch._inductor.config.patch({"combo_kernels_autotune": 0}):
            out, code = run_and_get_code(torch.compile(fn), *inps)
        self.assertEqual(out, fn(*inps))
        FileCheck().check("'num_kernels': 3").run(code[0])


@instantiate_parametrized_tests
class ComboKernelPDLTests(TestCase):
    # [CLASS DOC] Tests for Programmatic Dependent Launch (PDL) codegen in combo kernels.
    # setUp enables combo kernels, PDL, no benchmarking, unlimited distance and no
    # peak-memory limits.
    """Tests for PDL (Programmatic Dependent Launch) support in combo kernels."""

    def setUp(self):
        # [FUNCTION DOC] Reset metrics and push a config patch enabling combo kernels and PDL
        # onto an ExitStack.
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                    "triton.enable_pdl": True,
                    "combo_kernel_max_distance": -1,
                    "combo_kernel_peak_memory_increase_gb": None,
                    "combo_kernel_peak_memory_pct_threshold": None,
                }
            )
        )

    def tearDown(self):
        # [FUNCTION DOC] Close the patch stack, reset metrics, call base tearDown.
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    @skipIfRocm
    @unittest.skipIf(not SM90OrLater, "PDL requires SM90 or later (Hopper+)")
    def test_pdl_codegen_in_combo_kernel(self):
        # [FUNCTION DOC] Needs Hopper or newer, non-ROCm. Two same-sized pointwise ops compile
        # with round-robin dispatch; the code must set the PDL launch flag, and each subkernel
        # branch must have exactly one wait on the grid dependency before its load and one
        # launch-dependents call after, with no redundant waits between.
        """Test that PDL flag and gdc calls are generated in combo kernels."""

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return relu of a and sigmoid of b.
            return torch.relu(a), torch.sigmoid(b)

        inps = [
            torch.rand(1024, device=GPU_TYPE),
            torch.rand(1024, device=GPU_TYPE),
        ]

        fn_c = torch.compile(fn)
        _, code = run_and_get_code(fn_c, *inps)
        code = " ".join(code)

        # Check that launch_pdl is True and PDL API calls are generated
        FileCheck().check("'launch_pdl': True").run(code)

        # Each sub-kernel should have exactly one gdc_wait followed by one
        # gdc_launch_dependents, with no redundant waits in between.
        # Uses round-robin dispatch (pid % 2) since both tensors are same size.
        (
            FileCheck()
            .check("if pid")
            .check("tl.extra.cuda.gdc_wait()")
            .check("tl.load(")
            .check_not("tl.extra.cuda.gdc_wait()")
            .check("tl.extra.cuda.gdc_launch_dependents()")
            .check_not("tl.extra.cuda.gdc_wait()")
            .check("elif pid % 2 == 1:")
            .check("tl.extra.cuda.gdc_wait()")
            .check("tl.load(")
            .check_not("tl.extra.cuda.gdc_wait()")
            .check("tl.extra.cuda.gdc_launch_dependents()")
            .run(code)
        )

    @requires_gpu_and_triton
    @skipIfRocm
    @unittest.skipIf(not SM90OrLater, "PDL requires SM90 or later (Hopper+)")
    def test_pdl_combo_kernel_pointwise(self):
        # [FUNCTION DOC] Needs Hopper or newer, non-ROCm. Three pointwise ops compile to one
        # kernel with the right results; generated code has the PDL flag and each of the three
        # dispatch branches has one wait and one launch-dependents call, with no extra waits.
        """Test that pointwise combo kernels produce correct results with PDL."""

        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Return relu of a, sigmoid of b, tanh of c.
            return torch.relu(a), torch.sigmoid(b), torch.tanh(c)

        inps = [
            torch.rand(10, 10, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
            torch.rand(10, 10, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        code = " ".join(code)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

        # Verify combo kernel structure with PDL - each sub-kernel should have
        # exactly one gdc_wait and one gdc_launch_dependents, no redundant waits.
        (
            FileCheck()
            .check("'launch_pdl': True")
            .check("if pid < num_xblocks_0:")
            .check("tl.extra.cuda.gdc_wait()")
            .check_not("tl.extra.cuda.gdc_wait()")
            .check("tl.extra.cuda.gdc_launch_dependents()")
            .check("elif pid < num_xblocks_1:")
            .check("tl.extra.cuda.gdc_wait()")
            .check_not("tl.extra.cuda.gdc_wait()")
            .check("tl.extra.cuda.gdc_launch_dependents()")
            .check("elif pid < num_xblocks_2:")
            .check("tl.extra.cuda.gdc_wait()")
            .check_not("tl.extra.cuda.gdc_wait()")
            .check("tl.extra.cuda.gdc_launch_dependents()")
            .run(code)
        )

    @requires_gpu_and_triton
    @skipIfRocm
    @unittest.skipIf(not SM90OrLater, "PDL requires SM90 or later (Hopper+)")
    def test_pdl_combo_kernel_reduction(self):
        # [FUNCTION DOC] Needs Hopper or newer, non-ROCm. Reduction combo kernels with PDL
        # enabled: sum and mean over the last dim must match eager.
        """Test that reduction combo kernels produce correct results with PDL."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return last-dim sum of x and last-dim mean of y.
            return x.sum(dim=-1), y.mean(dim=-1)

        inps = [
            torch.rand(32, 1024, device=GPU_TYPE),
            torch.rand(32, 1024, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)


@instantiate_parametrized_tests
class ComboKernelTestsMaxAutotune(TestCase):
    # [CLASS DOC] Combo kernel tests under max autotune: setUp enables combo kernels with
    # per-subkernel blocks, max_autotune, no local autotune cache, unlimited distance and no
    # peak-memory limits. Tests look for debug logs from the sequential combo autotune path.
    def setUp(self):
        # [FUNCTION DOC] Reset metrics and push the max-autotune combo kernel config patch onto
        # an ExitStack.
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                    "combo_kernel_per_subkernel_blocks": True,
                    "max_autotune": True,
                    "autotune_local_cache": False,
                    "combo_kernel_max_distance": -1,
                    "combo_kernel_peak_memory_increase_gb": None,
                    "combo_kernel_peak_memory_pct_threshold": None,
                }
            )
        )

    def tearDown(self):
        # [FUNCTION DOC] Close the patch stack, reset metrics, call base tearDown.
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_gpu_and_triton
    def test_combo_kernel_max_autotune(self):
        # [FUNCTION DOC] Three pointwise activations of different shapes compile under debug
        # logging of the autotuner module; asserts at least one "Combo sequential autotune" log
        # line, eager-equal output and one generated kernel.
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Apply relu, sigmoid and tanh to three inputs and return
            # the results.
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(32, 1024, device=GPU_TYPE),
            torch.rand(64, 512, device=GPU_TYPE),
            torch.rand(16, 2048, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)

        logger = logging.getLogger("torch._inductor.runtime.triton_heuristics")
        with self.assertLogs(logger, level=logging.DEBUG) as cm:
            out_compiled, code = run_and_get_code(fn_c, *inps)
        chained_logs = [msg for msg in cm.output if "Combo sequential autotune" in msg]
        self.assertGreater(
            len(chained_logs),
            0,
            "_combo_sequential_autotune was not invoked",
        )
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    def test_combo_kernel_max_autotune_with_reduction(self):
        # [FUNCTION DOC] Same as the previous test with a sum and a mean reduction over the
        # last dim.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return last-dim sum of x and last-dim mean of y.
            return x.sum(dim=-1), y.mean(dim=-1)

        inps = [
            torch.rand(128, 256, device=GPU_TYPE),
            torch.rand(128, 256, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)

        logger = logging.getLogger("torch._inductor.runtime.triton_heuristics")
        with self.assertLogs(logger, level=logging.DEBUG) as cm:
            out_compiled, code = run_and_get_code(fn_c, *inps)
        chained_logs = [msg for msg in cm.output if "Combo sequential autotune" in msg]
        self.assertGreater(
            len(chained_logs),
            0,
            "_combo_sequential_autotune was not invoked",
        )
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    def test_combo_autotune_many_subkernels(self):
        # [FUNCTION DOC] Six pointwise ops over very different shapes compile to one kernel;
        # sequential autotune debug logs must appear and the output must match eager.
        def fn(a, b, c, d, e, f):
            # [FUNCTION DOC] HELPER: Return six different pointwise results of the six inputs.
            return (
                a * 2.0,
                b + 1.0,
                c.sin(),
                d.cos(),
                e.exp(),
                f.neg(),
            )

        inps = [
            torch.rand(8, 8192, device=GPU_TYPE),
            torch.rand(128, 64, device=GPU_TYPE),
            torch.rand(16, 4096, device=GPU_TYPE),
            torch.rand(512, 16, device=GPU_TYPE),
            torch.rand(32, 2048, device=GPU_TYPE),
            torch.rand(256, 32, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)

        logger = logging.getLogger("torch._inductor.runtime.triton_heuristics")
        with self.assertLogs(logger, level=logging.DEBUG) as cm:
            out_compiled, code = run_and_get_code(fn_c, *inps)

        chained_logs = [msg for msg in cm.output if "Combo sequential autotune" in msg]
        self.assertGreater(len(chained_logs), 0)
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    def test_combo_kernel_per_subkernel_reduction_hint(self):
        # [FUNCTION DOC] A last-dim sum and a dim 0 sum of same-shaped inputs. Reads the
        # generated code for per-subkernel reduction hint entries and asserts both exist with
        # value INNER for the first and OUTER for the second; output matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the last-dim sum of x and the dim 0 sum of y.
            return x.sum(dim=-1), y.sum(dim=0)

        inps = [
            torch.rand(128, 256, device=GPU_TYPE),
            torch.rand(128, 256, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        out, code = run_and_get_code(torch.compile(fn), *inps)
        self.assertEqual(out_eager, out)
        # Verify per-subkernel reduction hints in generated code
        found_hints = {}
        for c in code:
            for key in ["reduction_hint_0", "reduction_hint_1"]:
                m = re.search(rf"'{key}':\s*'(\w+)'", c)
                if m:
                    found_hints[key] = m.group(1)

        self.assertIn(
            "reduction_hint_0", found_hints, "Missing per-subkernel reduction_hint_0"
        )
        self.assertIn(
            "reduction_hint_1", found_hints, "Missing per-subkernel reduction_hint_1"
        )
        self.assertEqual(found_hints["reduction_hint_0"], "INNER")
        self.assertEqual(found_hints["reduction_hint_1"], "OUTER")

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        {
            "combo_kernel_autotune_grouping": True,
        }
    )
    def test_combo_autotune_grouping(self):
        # [FUNCTION DOC] With autotune grouping on, four pointwise ops fall into two size
        # buckets. Parses debug log lines naming phase one groups and asserts exactly two
        # distinct group indices, then checks output equals eager and one kernel is generated.
        def fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Return cos, sin, exp and negation of the four inputs.
            return a.cos(), b.sin(), c.exp(), d.neg()

        # a,b: numel=262144 -> bs=1024, c,d: numel=32 -> bs=256
        # Different bs -> different configs -> separate groups
        inps = [
            torch.rand(4, 65536, device=GPU_TYPE),
            torch.rand(4, 65536, device=GPU_TYPE),
            torch.rand(4, 8, device=GPU_TYPE),
            torch.rand(4, 8, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)

        logger = logging.getLogger("torch._inductor.runtime.triton_heuristics")
        with self.assertLogs(logger, level=logging.DEBUG) as cm:
            out_compiled, code = run_and_get_code(fn_c, *inps)

        # Parse "Phase 1 group N SK[...]" lines to check grouping
        group_lines = [
            msg for msg in cm.output if "Phase 1 group" in msg and "SK[" in msg
        ]
        group_indices = {
            int(re.search(r"group (\d+)", line).group(1))
            for line in group_lines
            if re.search(r"group (\d+)", line)
        }
        # 4 sub-kernels in 2 size buckets (rnumel 65536 vs 8) with identical
        # per-sub-kernel metadata within each bucket -> exactly 2 groups.
        self.assertEqual(
            len(group_indices),
            2,
            lambda msg: f"{msg}\nExpected 2 autotune groups, got {group_lines}",
        )
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_gpu_and_triton
    @torch._inductor.config.patch("combo_kernel_autotune_grouping", True)
    def test_combo_autotune_grouping_by_metadata_fingerprint(self):
        # [FUNCTION DOC] With autotune grouping on, builds combo grid metadata by hand for two
        # reduction subkernels and calls the per-subkernel-blocks handler with the reduction
        # heuristic mocked to return one fixed config. In each subtest one field of subkernel 1
        # is changed (size hints, load count, device TMA use, tiling scores) and the resulting
        # number of tuning groups stored in the metadata is compared: one group when identical,
        # two when any field differs.
        """Sub-kernels share a group iff every per-sub-kernel heuristic input
        is identical; any difference in the fingerprint separates them.
        """
        import triton

        def _sub_meta(
            num_load=1,
            num_store=1,
            num_reduction=1,
            autotune_hints=None,
            atomic_add_found=False,
            no_x_dim=False,
            uses_device_tma=False,
            tiling_scores=None,
        ):
            # [FUNCTION DOC] HELPER: Build a per-subkernel metadata mapping with the given
            # load, store and reduction counts, autotune hints (empty list if none), atomic
            # add flag, no-x-dim flag, device TMA flag and tiling scores (default x of 1 and
            # r0 of 8).
            return {
                "num_load": num_load,
                "num_store": num_store,
                "num_reduction": num_reduction,
                "autotune_hints": autotune_hints if autotune_hints is not None else [],
                "atomic_add_found": atomic_add_found,
                "no_x_dim": no_x_dim,
                "uses_device_tma": uses_device_tma,
                "tiling_scores": tiling_scores
                if tiling_scores is not None
                else {"x": 1, "r0_": 8},
            }

        def base_combo_meta():
            # [FUNCTION DOC] HELPER: Build combo grid metadata for two reduction subkernels
            # with identical size hints, INNER reduction hints and per-subkernel metadata, and
            # the grouping flag set.
            return {
                "num_kernels": 2,
                # Captured config — real combo_grid_meta() always sets this.
                "autotune_grouping": True,
                "heuristic_0": "reduction",
                "heuristic_1": "reduction",
                "size_hints_0": {"x": 2048, "r0_": 1024},
                "size_hints_1": {"x": 2048, "r0_": 1024},
                "reduction_hint_0": "INNER",
                "reduction_hint_1": "INNER",
                # Per-sub-kernel inductor_meta — the single source of truth
                # for fields fed into _subkernel_fingerprint.
                "inductor_meta_0": _sub_meta(),
                "inductor_meta_1": _sub_meta(),
            }

        def reduction_configs(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Mock replacement for the reduction heuristic; ignores its
            # arguments and returns a list with one Triton config (x block 8, reduction block
            # 256, 4 warps, 1 stage).
            return [
                triton.Config(
                    {"XBLOCK": 8, "R0_BLOCK": 256}, num_warps=4, num_stages=1
                ),
            ]

        # Each case mutates one field of sub-kernel 1 away from sub-kernel 0.
        # `None` means no mutation -> expect groups to merge.
        # For per-kernel fields, mutate inside inductor_meta_1 sub-dict.
        cases = [
            ("all identical merge", None, None, 1),
            ("different r0_numel", ("size_hints_1",), {"x": 2048, "r0_": 512}, 2),
            ("different num_load", ("inductor_meta_1", "num_load"), 12, 2),
            (
                "different uses_device_tma",
                ("inductor_meta_1", "uses_device_tma"),
                True,
                2,
            ),
            (
                "different tiling_scores",
                ("inductor_meta_1", "tiling_scores"),
                {"x": 8, "r0_": 1},
                2,
            ),
        ]

        for desc, path, value, expected_groups in cases:
            with self.subTest(desc):
                meta = base_combo_meta()
                if path is not None:
                    if len(path) == 1:
                        meta[path[0]] = value
                    else:
                        meta[path[0]][path[1]] = value
                inductor_meta = {"combo_grid_meta": meta}

                with unittest.mock.patch(
                    "torch._inductor.runtime.triton_heuristics.reduction",
                    side_effect=reduction_configs,
                ):
                    torch._inductor.runtime.triton_heuristics._handle_combo_kernel_per_subkernel_blocks(
                        {"x": 2048, "r0_": 1024},
                        inductor_meta,
                        triton_meta={},
                    )

                groups = inductor_meta["combo_tuning_groups"]
                self.assertEqual(
                    len(groups),
                    expected_groups,
                    lambda msg: f"{msg}\n{desc}: expected {expected_groups} group(s), got {len(groups)}",
                )

    @requires_gpu_and_triton
    @parametrize("per_subkernel", [True, False])
    def test_combo_max_persistent_rblock_gated_on_per_subkernel(self, per_subkernel):
        # [FUNCTION DOC] Parametrized by per_subkernel. Two row sums of different widths
        # compile with the option set to the parameter; output matches eager. When
        # per-subkernel blocks are on, the code must not carry the max persistent reduction
        # block key; otherwise it must carry it with 1024.
        """The combo-only HIP filter (XBLOCK * max_persistent_rblock <= 4096)
        from PR #175671 guards a shared-XBLOCK blowup that only exists in
        the legacy combo path. Under per_subkernel_blocks=True each sub has
        its own XBLOCK_n so the filter must be skipped (key absent). Under
        per_subkernel_blocks=False the combo-max value must be set as before.
        """

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return the last-dim sums of both inputs.
            return a.sum(-1), b.sum(-1)

        inps = [
            torch.rand(32, 256, device=GPU_TYPE),
            torch.rand(32, 1024, device=GPU_TYPE),
        ]
        with torch._inductor.config.patch(
            {"combo_kernel_per_subkernel_blocks": per_subkernel}
        ):
            out_eager = fn(*inps)
            out_compiled, code = run_and_get_code(torch.compile(fn), *inps)
        self.assertEqual(out_eager, out_compiled)
        joined = " ".join(code)
        if per_subkernel:
            self.assertNotIn("'max_persistent_rblock'", joined)
        else:
            self.assertIn("'max_persistent_rblock': 1024", joined)

    @requires_gpu_and_triton
    def test_combo_kernel_coordesc_tunes_largest_subkernel_first(self):
        # [FUNCTION DOC] With coordinate descent tuning on, compiles three activations of
        # different sizes while capturing debug logs of the coordinate descent tuner. Output
        # must equal eager; there must be baseline and attempt logs naming per-subkernel block
        # fields, more than one distinct block configuration tried, and the first attempt must
        # change only the block field of subkernel 1 (the largest one), i.e. the largest
        # subkernel is tuned first.
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Return relu, sigmoid and tanh of the three inputs.
            return (
                torch.nn.functional.relu(a),
                torch.nn.functional.sigmoid(b),
                torch.nn.functional.tanh(c),
            )

        inps = [
            torch.rand(32, 1024, device=GPU_TYPE),
            torch.rand(256, 256, device=GPU_TYPE),
            torch.rand(16, 128, device=GPU_TYPE),
        ]

        out_eager = fn(*inps)

        def parse_block_cfg(msg: str) -> dict[str, int]:
            # [FUNCTION DOC] HELPER: Parse a log message into a mapping from per-subkernel
            # block names to integer values, taking every name-and-number pair that matches the
            # block naming pattern.
            return {
                m.group(1): int(m.group(2))
                for m in re.finditer(r"(\w+BLOCK_\d+): (\d+)", msg)
            }

        logger = logging.getLogger("torch._inductor.runtime.coordinate_descent_tuner")
        with torch._inductor.config.patch(coordinate_descent_tuning=True):
            with self.assertLogs(logger, level=logging.DEBUG) as cm:
                out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

        baseline_log = next(
            msg for msg in cm.output if "Baseline Config" in msg and "XBLOCK_" in msg
        )
        baseline_cfg = parse_block_cfg(baseline_log)
        try_logs = [
            msg for msg in cm.output if "Try config" in msg and "XBLOCK_" in msg
        ]
        self.assertGreater(
            len(try_logs), 0, "Coordinate descent did not try combo fields"
        )
        distinct_block_cfgs = {
            tuple(sorted(parse_block_cfg(msg).items())) for msg in try_logs
        }
        self.assertGreater(
            len(distinct_block_cfgs),
            1,
            "Coordinate descent did not explore different suffixed block sizes.",
        )

        first_cfg = parse_block_cfg(try_logs[0])
        changed_fields = {
            key for key, value in first_cfg.items() if baseline_cfg.get(key) != value
        }
        self.assertEqual(
            changed_fields,
            {"XBLOCK_1"},
            lambda msg: f"{msg}\nExpected the first combo coordesc step to tune the largest subkernel first, got {changed_fields}",
        )


@instantiate_parametrized_tests
class ComboKernelMetadataTests(TestCase):
    # [CLASS DOC] Tests that combo kernel generated code carries the expected metadata
    # (inductor_meta, triton meta, per-subkernel fields, profiling numbers). setUp
    # enables combo kernels with per-subkernel blocks and no benchmarking; _combo_code compiles
    # and returns code text.
    def setUp(self):
        # [FUNCTION DOC] Reset metrics and push a config patch (combo kernels, no benchmarking,
        # per-subkernel blocks) onto an ExitStack.
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                    "combo_kernel_per_subkernel_blocks": True,
                }
            )
        )

    def tearDown(self):
        # [FUNCTION DOC] Close the patch stack, reset metrics, call base tearDown.
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    def _combo_code(self, fn, inps):
        # [FUNCTION DOC] HELPER: Run the function eagerly and compiled, assert the outputs are
        # equal, and return all generated code strings joined with spaces. Args: the function
        # and the input list.
        out_eager = fn(*inps)
        out_compiled, code = run_and_get_code(torch.compile(fn), *inps)
        self.assertEqual(out_eager, out_compiled)
        return " ".join(code)

    @requires_gpu_and_triton
    def test_combo_inductor_meta_has_optimize_mem(self):
        # [FUNCTION DOC] Two pointwise ops compile to a combo whose code includes the memory
        # optimization flag set true.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return relu of a and sigmoid of b.
            return torch.relu(a), torch.sigmoid(b)

        inps = [torch.rand(1024, device=GPU_TYPE) for _ in range(2)]
        code = self._combo_code(fn, inps)
        self.assertIn("'optimize_mem': True", code)

    @requires_gpu_and_triton
    def test_combo_inductor_meta_optimize_mem_false_in_training_forward(self):
        # [FUNCTION DOC] Same function with inputs requiring grad (training forward): the
        # memory optimization flag in the generated code must be false.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return relu of a and sigmoid of b.
            return torch.relu(a), torch.sigmoid(b)

        inps = [torch.rand(1024, device=GPU_TYPE, requires_grad=True) for _ in range(2)]
        code = self._combo_code(fn, inps)
        self.assertIn("'optimize_mem': False", code)

    @requires_gpu_and_triton
    @parametrize("disable_ftz", [False, True])
    def test_combo_triton_meta_has_disable_ftz(self, disable_ftz):
        # [FUNCTION DOC] Parametrized by disable_ftz. With the numerics config set accordingly,
        # the generated combo kernel meta must contain the disable flush-to-zero entry with the
        # same value.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return relu of a and sigmoid of b.
            return torch.relu(a), torch.sigmoid(b)

        inps = [torch.rand(1024, device=GPU_TYPE) for _ in range(2)]
        with torch._inductor.config.patch({"eager_numerics.disable_ftz": disable_ftz}):
            code = self._combo_code(fn, inps)
        self.assertIn(f"'disable_ftz': {disable_ftz}", code)

    @requires_gpu_and_triton
    def test_combo_pointwise_combo_grid_meta_has_per_subkernel_fields(self):
        # [FUNCTION DOC] Three pointwise subkernels: the generated code must have, for each
        # subkernel index, a per-subkernel inductor meta entry followed by load, store,
        # reduction counts, autotune hints and atomic add fields (order among the fields is
        # not enforced).
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Return relu, sigmoid and tanh of the three inputs.
            return torch.relu(a), torch.sigmoid(b), torch.tanh(c)

        inps = [torch.rand(1024, device=GPU_TYPE) for _ in range(3)]
        code = self._combo_code(fn, inps)

        # Each sub-kernel's inductor_meta_{n} sub-dict must contain the
        # per-kernel autotune fields.
        fc = FileCheck()
        for num in range(3):
            fc = fc.check(f"'inductor_meta_{num}'")
            for field in (
                "num_load",
                "num_store",
                "num_reduction",
                "autotune_hints",
                "atomic_add_found",
            ):
                fc = fc.check_dag(f"'{field}'")
        fc.run(code)

    @requires_gpu_and_triton
    def test_combo_reduction_combo_grid_meta_has_per_subkernel_fields(self):
        # [FUNCTION DOC] Same field check for two reduction subkernels, covering load, store
        # and reduction counts.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return the last-dim sum of a and mean of b.
            return a.sum(-1), b.mean(-1)

        inps = [torch.rand(32, 1024, device=GPU_TYPE) for _ in range(2)]
        code = self._combo_code(fn, inps)

        fc = FileCheck()
        for num in range(2):
            fc = fc.check(f"'inductor_meta_{num}'")
            for field in ("num_load", "num_store", "num_reduction"):
                fc = fc.check_dag(f"'{field}'")
        fc.run(code)

    @requires_gpu_and_triton
    @torch._inductor.config.patch({"deterministic": True})
    def test_combo_reduction_deterministic_has_contiguous_rdim_per_subkernel(self):
        # [FUNCTION DOC] In deterministic mode, each reduction subkernel meta must carry the
        # entry saying whether it has a load or store with contiguous reduction dimension.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return the last-dim sum of a and mean of b.
            return a.sum(-1), b.mean(-1)

        inps = [torch.rand(32, 1024, device=GPU_TYPE) for _ in range(2)]
        code = self._combo_code(fn, inps)

        fc = FileCheck()
        for num in range(2):
            fc = fc.check(f"'inductor_meta_{num}'").check(
                "'has_loadstore_with_contiguous_rdim'"
            )
        fc.run(code)

    @requires_gpu_and_triton
    def test_combo_per_kernel_inductor_meta_matches_standalone(self):
        # [FUNCTION DOC] Extracts selected per-kernel meta fields (load/store/reduction counts,
        # atomic add, no-x-dim, autotune hints, tiling scores) by regex from kernels compiled
        # with combo kernels off (one dict per standalone kernel, split on the meta keyword)
        # and from per-subkernel meta entries of the combo kernel, and asserts both lists of
        # dicts are equal.
        per_kernel_fields = re.compile(
            r"'(num_load|num_store|num_reduction|"
            r"atomic_add_found|no_x_dim|"
            r"autotune_hints|tiling_scores)'\s*:\s*"
            r"(\d+|True|False|None|\{[^{}]*\}|set\([^)]*\))"
        )

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return relu of a and sigmoid of b.
            return torch.relu(a), torch.sigmoid(b)

        inps = [torch.rand(1024, device=GPU_TYPE) for _ in range(2)]

        # Standalone: one wrapper source can contain multiple kernels; split
        # by "inductor_meta=" so each chunk represents one kernel.
        with torch._inductor.config.patch({"combo_kernels": False}):
            torch._dynamo.reset()
            _, standalone_codes = run_and_get_code(torch.compile(fn), *inps)
        standalone = [
            dict(per_kernel_fields.findall(chunk))
            for c in standalone_codes
            for chunk in c.split("inductor_meta=")[1:]
        ]

        # Combo: per-sub-kernel inductor_meta_{i} sub-dicts inside combo_grid_meta.
        # The inner regex allows one level of `{...}` nesting (for tiling_scores).
        code = self._combo_code(fn, inps)
        combo = [
            dict(per_kernel_fields.findall(s))
            for s in re.findall(
                r"'inductor_meta_\d+': (\{(?:[^{}]|\{[^{}]*\})*\})", code
            )
        ]
        self.assertEqual(standalone, combo)

    @requires_gpu_and_triton
    @torch._inductor.config.patch({"benchmark_kernel": True})
    def test_combo_kernel_num_gb_and_flop_match_standalone_sum(self):
        # [FUNCTION DOC] With kernel benchmarking on, compares the combo kernel's top-level
        # memory traffic and flop meta values against the sums over standalone kernels compiled
        # with combo off. The last match in the combo source is taken as the combo-level value;
        # compared to six places.
        kernel_num_gb_re = re.compile(r"'kernel_num_gb'\s*:\s*([\d.eE+-]+)")
        kernel_flop_re = re.compile(r"'kernel_flop'\s*:\s*([\d.eE+-]+)")

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return relu of a and sigmoid of b.
            return torch.relu(a), torch.sigmoid(b)

        inps = [torch.rand(1024, device=GPU_TYPE) for _ in range(2)]

        with torch._inductor.config.patch({"combo_kernels": False}):
            torch._dynamo.reset()
            _, sa_codes = run_and_get_code(torch.compile(fn), *inps)
        sa_text = " ".join(sa_codes)
        sa_gb_sum = sum(float(m.group(1)) for m in kernel_num_gb_re.finditer(sa_text))
        sa_flop_sum = sum(float(m.group(1)) for m in kernel_flop_re.finditer(sa_text))

        combo_code = self._combo_code(fn, inps)
        # Combo source contains kernel_num_gb both inside each per-sub-kernel
        # inductor_meta_{i} sub-dict and at the combo-level (the sum). The
        # outer combo-level value is inserted AFTER combo_grid_meta in dict
        # order, so the last match is the combo-level sum.
        combo_gb_matches = kernel_num_gb_re.findall(combo_code)
        combo_flop_matches = kernel_flop_re.findall(combo_code)
        self.assertTrue(combo_gb_matches, "combo source missing kernel_num_gb")
        self.assertTrue(combo_flop_matches, "combo source missing kernel_flop")

        self.assertAlmostEqual(float(combo_gb_matches[-1]), sa_gb_sum, places=6)
        self.assertAlmostEqual(float(combo_flop_matches[-1]), sa_flop_sum, places=6)

    @requires_gpu_and_triton
    @torch._inductor.config.patch({"profile_bandwidth": True})
    def test_combo_inductor_meta_has_kernel_num_gb_under_profile_bandwidth(self):
        # [FUNCTION DOC] Under bandwidth profiling, the combo meta has the memory traffic entry
        # but no flop entry.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return relu of a and sigmoid of b.
            return torch.relu(a), torch.sigmoid(b)

        inps = [torch.rand(1024, device=GPU_TYPE) for _ in range(2)]
        code = self._combo_code(fn, inps)
        self.assertIn("'kernel_num_gb'", code)
        self.assertNotIn("'kernel_flop'", code)

    @requires_gpu_and_triton
    def test_combo_inductor_meta_no_kernel_num_gb_without_profile(self):
        # [FUNCTION DOC] Without profiling or benchmarking, the combo code has neither memory
        # traffic nor flop meta.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return relu of a and sigmoid of b.
            return torch.relu(a), torch.sigmoid(b)

        inps = [torch.rand(1024, device=GPU_TYPE) for _ in range(2)]
        code = self._combo_code(fn, inps)
        self.assertNotIn("'kernel_num_gb'", code)
        self.assertNotIn("'kernel_flop'", code)

    @requires_gpu_and_triton
    def test_combo_rms_norm_forwards_add_persistent_rblock(self):
        # [FUNCTION DOC] Two parallel RMS norms over large rows form a combo; each
        # per-subkernel meta must include the add-persistent-rblock entry (forwarded from the
        # standalone large-rblock optimization so the specialized config can be emitted).
        """rms_norm at (2048, 4096) triggers the large-rblock persistent
        optimization for standalone kernels (set by triton.py:5820). Two
        parallel rms_norms forming a combo kernel must forward
        add_persistent_rblock per sub-kernel so _reduction_configs can emit
        the same specialized large-RBLOCK config."""

        def rms_norm(x, w, eps=1e-6):
            # [FUNCTION DOC] HELPER: RMS norm: weight times x times reciprocal sqrt of (mean of
            # squares over the last dim plus eps).
            v = x.pow(2).mean(-1, keepdim=True)
            return w * x * torch.rsqrt(v + eps)

        def fn(a, wa, b, wb):
            # [FUNCTION DOC] HELPER: Apply the RMS norm helper to two input and weight pairs;
            # return both.
            return rms_norm(a, wa), rms_norm(b, wb)

        a = torch.rand(2048, 4096, device=GPU_TYPE)
        b = torch.rand(2048, 4096, device=GPU_TYPE)
        wa = torch.rand(4096, device=GPU_TYPE)
        wb = torch.rand(4096, device=GPU_TYPE)
        code = self._combo_code(fn, [a, wa, b, wb])

        # add_persistent_rblock must appear inside each sub-kernel's
        # inductor_meta_{n} sub-dict (single source of truth via
        # TritonKernel.inductor_meta_per_kernel).
        fc = FileCheck()
        for num in range(2):
            fc = fc.check(f"'inductor_meta_{num}'").check("'add_persistent_rblock'")
        fc.run(code)

    @requires_gpu_and_triton
    @torch._inductor.config.patch({"benchmark_combo_kernel": True})
    def test_benchmark_combo_kernel_emits_real_num_gb(self):
        # [FUNCTION DOC] With combo benchmarking on, the generated benchmark code must print a
        # real decimal bandwidth number and quote the device string in the benchmark setup
        # (not unquoted).
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return relu of a and sigmoid of b.
            return torch.relu(a), torch.sigmoid(b)

        inps = [torch.rand(1024, device=GPU_TYPE) for _ in range(2)]
        code = self._combo_code(fn, inps)
        self.assertRegex(code, r"num_gb = \d*\.\d+")
        self.assertIn(f"device='{GPU_TYPE}'", code)
        self.assertNotIn(f"device={GPU_TYPE}", code)


# Minimal scheduler doubles for direct _try_combo_with_memory_check tests.
class _PeakMemFakeNode:
    # [CLASS DOC] HELPER: Fake scheduler node for combo memory check tests. Holds a name, a
    # bare scheduler object, an output list, a mem-planning node with no predecessor buffers,
    # and an optional list of sub-nodes (when set, outputs come from the sub-nodes).
    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Store the name and set up the empty output list, scheduler
        # object, planning node with an empty predecessor set, and no sub-nodes.
        self.name = name
        self.scheduler = object()
        self._outputs: list = []
        self.mpi_node = SimpleNamespace(pred_buffers=set())
        self.snodes: list | None = None

    def get_name(self) -> str:
        # [FUNCTION DOC] HELPER: Return the node name.
        return self.name

    def get_first_name(self) -> str:
        # [FUNCTION DOC] HELPER: Return the node name (same as the full name here).
        return self.name

    def get_outputs(self):
        # [FUNCTION DOC] HELPER: Return the concatenated outputs of the sub-nodes if present,
        # else the own outputs.
        if self.snodes is not None:
            out = []
            for s in self.snodes:
                out.extend(s.get_outputs())
            return out
        return self._outputs


class _PeakMemFakeBuffer:
    # [CLASS DOC] HELPER: Fake buffer with a name and a planning record holding allocation
    # size, free size and successor nodes.
    def __init__(self, name: str, succ_nodes, size_alloc: int, size_free: int) -> None:
        # [FUNCTION DOC] HELPER: Store name and build the planning record from sizes
        # and successors.
        self.name = name
        self.mpi_buffer = SimpleNamespace(
            size_alloc=size_alloc,
            size_free=size_free,
            succ_nodes=succ_nodes,
        )

    def get_name(self) -> str:
        # [FUNCTION DOC] HELPER: Return the buffer name.
        return self.name


class _PeakMemFakeScheduler:
    # [CLASS DOC] HELPER: Fake scheduler exposing a node list, a map from names to fused nodes
    # (empty if not given), and an identity topological sort.
    def __init__(self, nodes, name_to_fused_node=None) -> None:
        # [FUNCTION DOC] HELPER: Store the node list and the name map (empty mapping
        # when None).
        self.nodes = nodes
        self.name_to_fused_node = (
            {} if name_to_fused_node is None else name_to_fused_node
        )

    def topological_sort_schedule(self, nodes):
        # [FUNCTION DOC] HELPER: Return the given nodes unchanged.
        return nodes


@instantiate_parametrized_tests
class ComboKernelPeakMemoryTests(InductorTestCase):
    # [CLASS DOC] Tests for memory-aware acceptance and commit of combo kernels (peak memory
    # thresholds, reordering). setUp enables combo kernels with per-subkernel blocks; helper
    # methods build thresholds and a WideResNet-like model.
    """Coverage for memory-aware combo-kernel acceptance and commit logic."""

    def setUp(self):
        # [FUNCTION DOC] Reset metrics and push a config patch (combo kernels, no benchmarking,
        # per-subkernel blocks) onto an ExitStack.
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                    "combo_kernel_per_subkernel_blocks": True,
                }
            )
        )

    def tearDown(self):
        # [FUNCTION DOC] Close the patch stack, reset metrics, call base tearDown.
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @staticmethod
    def _thresholds(*, abs_thr_gb=None, pct_thr=None, max_distance=-1):
        # [FUNCTION DOC] HELPER: Static method; returns a config mapping with the peak memory
        # absolute and percent thresholds and the max distance, keyword-only with defaults
        # none, none and -1.
        return {
            "combo_kernel_peak_memory_increase_gb": abs_thr_gb,
            "combo_kernel_peak_memory_pct_threshold": pct_thr,
            "combo_kernel_max_distance": max_distance,
        }

    @requires_cuda_and_triton
    @parametrize("gate_enabled", [True, False])
    def test_peak_memory_reorder_order(self, gate_enabled):
        # [FUNCTION DOC] CUDA only, parametrized by whether the percent threshold gate is on.
        # Wraps combo node creation, reorder-for-peak-memory and planning-info preparation
        # with recorders, compiles two pointwise ops in a fresh cache and asserts the result
        # matches eager, the call order is reorder then combo, and planning info is prepared
        # exactly once.
        from torch._inductor import memory
        from torch._inductor.scheduler import Scheduler

        calls = []
        planning_calls = 0
        create_combo = Scheduler.create_combo_kernel_nodes
        prepare_planning_info = memory.prepare_planning_info
        reorder = memory.reorder_for_peak_memory

        def record_combo(scheduler, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Record a combo event then delegate to the original combo
            # node creation.
            calls.append("combo")
            return create_combo(scheduler, *args, **kwargs)

        def record_reorder(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Record a reorder event then delegate to the original
            # reorder function.
            calls.append("reorder")
            return reorder(*args, **kwargs)

        def record_prepare_planning_info(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Count the call (nonlocal counter) and delegate to the
            # original planning-info preparation.
            nonlocal planning_calls
            planning_calls += 1
            return prepare_planning_info(*args, **kwargs)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return sine of a and cosine of b.
            return a.sin(), b.cos()

        inputs = [torch.randn(16, device=GPU_TYPE) for _ in range(2)]
        with (
            patch.object(Scheduler, "create_combo_kernel_nodes", record_combo),
            patch.object(memory, "prepare_planning_info", record_prepare_planning_info),
            patch.object(memory, "reorder_for_peak_memory", record_reorder),
            fresh_cache(),
            torch._inductor.config.patch(
                {
                    "combo_kernel_peak_memory_pct_threshold": (
                        0.05 if gate_enabled else None
                    ),
                    "combo_kernel_peak_memory_increase_gb": None,
                }
            ),
        ):
            self.assertEqual(torch.compile(fn)(*inputs), fn(*inputs))

        self.assertEqual(calls, ["reorder", "combo"])
        self.assertEqual(planning_calls, 1)

    @staticmethod
    def _make_wide_resnet_like():
        # [FUNCTION DOC] HELPER: Static method that builds and returns a small WideResNet-like
        # module: a stem, four layers of bottleneck blocks and a classifier (details in the
        # nested definitions).
        """Build a WideResNet-like model."""

        class Bottleneck(torch.nn.Module):
            # [CLASS DOC] HELPER: Residual bottleneck block with three convolutions, each
            # followed by batch norm, in-place relu, optional downsample on the shortcut,
            # expansion factor 4.
            expansion = 4

            def __init__(self, in_ch, mid_ch, stride=1, downsample=None):
                # [FUNCTION DOC] HELPER: Create the three conv and batch norm pairs (channel
                # counts in, mid, mid times expansion; stride on the middle conv), relu and the
                # optional downsample module.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_ch, mid_ch, 1, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(mid_ch)
                self.conv2 = torch.nn.Conv2d(
                    mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False
                )
                self.bn2 = torch.nn.BatchNorm2d(mid_ch)
                self.conv3 = torch.nn.Conv2d(
                    mid_ch, mid_ch * self.expansion, 1, bias=False
                )
                self.bn3 = torch.nn.BatchNorm2d(mid_ch * self.expansion)
                self.relu = torch.nn.ReLU(inplace=True)
                self.downsample = downsample

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Run the three conv-bn stages with relu after the first
                # two, add the (possibly downsampled) input and apply relu.
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                if self.downsample is not None:
                    identity = self.downsample(x)
                return self.relu(out + identity)

        def make_layer(in_ch, mid_ch, blocks, stride=1):
            # [FUNCTION DOC] HELPER: Build a sequential layer: first block with a conv and
            # batch norm downsample shortcut, then the remaining blocks at expanded width.
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, mid_ch * 4, 1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(mid_ch * 4),
            )
            layers = [Bottleneck(in_ch, mid_ch, stride, downsample)]
            for _ in range(1, blocks):
                layers.append(Bottleneck(mid_ch * 4, mid_ch))
            return torch.nn.Sequential(*layers)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Small bottleneck-ResNet-like network: 7x7 stride-2 stem with
            # batch norm, relu and max pool, three layers of bottleneck blocks (3, 4 and 20
            # blocks), global average pool and a 1000-way linear classifier.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the stem, pooling, the three bottleneck layers
                # via the layer builder, the average pool and the classifier.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
                self.bn1 = torch.nn.BatchNorm2d(64)
                self.relu = torch.nn.ReLU(inplace=True)
                self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
                self.layer1 = make_layer(64, 128, blocks=3)
                self.layer2 = make_layer(512, 256, blocks=4, stride=2)
                self.layer3 = make_layer(1024, 512, blocks=20, stride=2)
                self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
                self.fc = torch.nn.Linear(2048, 1000)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Run stem, max pool, the three layers, average pool,
                # flatten and classifier.
                x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                return self.fc(self.avgpool(x).flatten(1))

        return Model()

    @staticmethod
    def _try_combo_with_fake_scheduler(
        nodes,
        group_nodes,
        *,
        baseline_peak,
        baseline_live_before,
        thresholds,
        graph_outputs=None,
    ):
        # [FUNCTION DOC] HELPER: Static method running the scheduler memory-gated combo attempt
        # with a fake scheduler. Builds a memory context (graph outputs default to empty, index
        # per node, baseline and running peak, baseline live memory), patches combo node
        # construction with a fake that unions the members' predecessor buffers, applies the
        # threshold config, and returns the pair of combo node (or none if rejected) and its
        # step from the scheduler method, with autotune off.
        from torch._inductor.scheduler import ComboKernelMemoryContext, Scheduler

        scheduler = _PeakMemFakeScheduler(nodes)
        mem_ctx = ComboKernelMemoryContext(
            graph_outputs=set() if graph_outputs is None else graph_outputs,
            node_to_idx={node: idx for idx, node in enumerate(nodes)},
            baseline_peak=baseline_peak,
            running_peak=baseline_peak,
            baseline_live_before=baseline_live_before,
        )

        def _fake_combo(scheduler_arg, snodes, **kwargs):
            # [FUNCTION DOC] HELPER: Fake combo node factory: builds a fake node named combo
            # whose sub-nodes are the members and whose planning node has the union of the
            # members' predecessor buffers.
            n = _PeakMemFakeNode("combo")
            n.snodes = list(snodes)
            pred_buffers: set = set()
            for m in snodes:
                pred_buffers.update(m.mpi_node.pred_buffers)
            n.mpi_node = SimpleNamespace(pred_buffers=pred_buffers)
            return n

        with (
            patch(
                "torch._inductor.scheduler.ForeachKernelSchedulerNode",
                _fake_combo,
            ),
            torch._inductor.config.patch(**thresholds),
        ):
            return Scheduler._try_combo_with_memory_check(
                scheduler,
                group_nodes,
                mem_ctx,
                enable_autotune=False,
            )

    def test_threshold_gating(self):
        # [FUNCTION DOC] Four fake nodes with 2 MB buffers freed by consumers; the combo of the
        # two producers forces a 2 MB peak increase. Absolute limit zero, percent limit zero
        # and a 1 MB absolute limit must each reject (no combo); a one GB limit must accept
        # with combo step 0.
        """abs_thr/pct_thr set to 0 or a too-small bound reject."""
        ONE_MB = 1024 * 1024
        a = _PeakMemFakeNode("a")
        consume_a = _PeakMemFakeNode("consume_a")
        b = _PeakMemFakeNode("b")
        consume_b = _PeakMemFakeNode("consume_b")
        nodes = [a, consume_a, b, consume_b]
        # Each buffer is 2 MB so the combo forces a +2 MB peak delta —
        # large enough to test against MB-scale thresholds.
        buf_a = _PeakMemFakeBuffer("buf_a", {consume_a}, 2 * ONE_MB, 2 * ONE_MB)
        buf_b = _PeakMemFakeBuffer("buf_b", {consume_b}, 2 * ONE_MB, 2 * ONE_MB)
        a._outputs = [buf_a]
        b._outputs = [buf_b]
        consume_a.mpi_node.pred_buffers = {buf_a}
        consume_b.mpi_node.pred_buffers = {buf_b}
        # Baseline: a allocs (peak 2 MB), consume_a frees, b allocs (peak 2 MB),
        # consume_b frees. baseline_live_before tracks live bytes before each step.
        baseline_live_before = [0, 2 * ONE_MB, 0, 2 * ONE_MB, 0]

        def run(thresholds):
            # [FUNCTION DOC] HELPER: Run the fake-scheduler combo attempt on the producers a
            # and b with the given thresholds and the shared baseline numbers.
            return self._try_combo_with_fake_scheduler(
                nodes,
                [a, b],
                baseline_peak=2 * ONE_MB,
                baseline_live_before=baseline_live_before,
                thresholds=thresholds,
            )

        # Rejection cases: any limit below the +2 MB delta the combo forces.
        for label, thresholds in (
            ("abs_gb=0", self._thresholds(abs_thr_gb=0.0)),
            ("pct=0", self._thresholds(pct_thr=0.0)),
            ("abs_gb=1MB", self._thresholds(abs_thr_gb=1.0 / 1024)),
        ):
            combo, _ = run(thresholds)
            self.assertIsNone(combo, lambda msg: f"{msg}\n{label} should reject")

        combo, combo_step = run(self._thresholds(abs_thr_gb=1.0))
        self.assertIsNotNone(combo)
        self.assertEqual(combo_step, 0)

    def test_region_carry_in_uses_post_free_boundary(self):
        # [FUNCTION DOC] Region boundary accounting: a buffer freed before the region starts
        # (producer a, freed at step 1) must not count as carry-in memory for a region starting
        # at step 2 of combining c and d. With a zero absolute limit and graph outputs as the
        # two new buffers, the combo must be accepted.
        a = _PeakMemFakeNode("a")
        consume_a = _PeakMemFakeNode("consume_a")
        c = _PeakMemFakeNode("c")
        d = _PeakMemFakeNode("d")
        nodes = [a, consume_a, c, d]

        buf_a = _PeakMemFakeBuffer("buf_a", {consume_a}, 100, 100)
        buf_c = _PeakMemFakeBuffer("buf_c", set(), 100, 100)
        buf_d = _PeakMemFakeBuffer("buf_d", set(), 100, 100)
        a._outputs = [buf_a]
        c._outputs = [buf_c]
        d._outputs = [buf_d]
        consume_a.mpi_node.pred_buffers = {buf_a}

        # buf_a is freed at step 1, so a region starting at step 2 has no
        # carry-in from buf_a.
        baseline_live_before = [0, 100, 0, 100, 200]

        combo, _ = self._try_combo_with_fake_scheduler(
            nodes,
            [c, d],
            baseline_peak=200,
            baseline_live_before=baseline_live_before,
            thresholds=self._thresholds(abs_thr_gb=0.0, pct_thr=None),
            graph_outputs={"buf_c", "buf_d"},
        )
        self.assertIsNotNone(combo)

    @skipIfRocm  # https://github.com/pytorch/pytorch/issues/182444
    @requires_cuda_and_triton
    def test_combo_kernel_peak_memory_wide_resnet(self):
        # [FUNCTION DOC] CUDA only, skipped on ROCm. Builds a WideResNet-like model, compiles
        # it twice in fresh caches (with and without a tight 1 MB peak memory threshold, same
        # max distance) and measures runtime peak CUDA memory of the second forward call;
        # asserts the tight threshold gives a strictly lower peak than gating disabled.
        """A tight peak-memory threshold must measurably reduce the
        runtime CUDA peak memory of the compiled forward pass compared
        to the gating-disabled baseline. Both runs pin
        combo_kernel_max_distance so the windowing behavior is identical
        and the only difference is whether the gate rejects oversized
        combos."""
        model = ComboKernelPeakMemoryTests._make_wide_resnet_like().to(GPU_TYPE).eval()
        x = torch.randn(1, 3, 224, 224, device=GPU_TYPE)

        def compile_and_measure_peak(**cfg):
            # [FUNCTION DOC] HELPER: Reset dynamo and metrics, clear CUDA cache and peak stats,
            # compile the model under the given config patch in a fresh cache, run it once to
            # warm up, reset peak stats, run again under no-grad and return the peak allocated
            # CUDA memory in bytes.
            torch._dynamo.reset()
            torch._inductor.metrics.reset()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with (
                fresh_cache(),
                torch._inductor.config.patch(
                    **cfg,
                ),
            ):
                compiled = torch.compile(model)
                with torch.no_grad():
                    compiled(x)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    compiled(x)
                torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated()

        # Gating disabled: combos can co-allocate freely -> higher peak.
        peak_disabled = compile_and_measure_peak(
            **self._thresholds(abs_thr_gb=None, pct_thr=None, max_distance=32),
        )
        # Tight abs threshold (1 MB -> ~0.001 GB): reject combos that
        # would inflate peak.
        peak_tight = compile_and_measure_peak(
            **self._thresholds(abs_thr_gb=1.0 / 1024, max_distance=32),
        )
        self.assertLess(
            peak_tight,
            peak_disabled,
            lambda msg: f"{msg}\ntight threshold did not reduce runtime peak memory "
            f"(tight={peak_tight}, disabled={peak_disabled})",
        )

    def test_estimate_region_peak_memory(self):
        # [FUNCTION DOC] Checks the region peak memory estimator on fake nodes with steps, one
        # buffer freed in-window, one graph output never freed, and one allocation past the
        # window (ignored). The walk allocates and checks the peak before freeing at each step;
        # expects peak of 350.
        from torch._inductor import memory as mem_mod

        # Window [0, 5] with three buffers + one out-of-window node:
        #   bufA: size 100, alloc at step 1 (a1), last use step 3 (b3)
        #   bufB: size 200, alloc at step 2 (a2), last use step 5 (b5)
        #   bufC: size  50, alloc at step 3 (a3), graph output (never freed)
        #   bufD: size 999, alloc at step 100 (a100) — past window, must skip
        a1, a2, a3, a100, b3, b5 = (
            _PeakMemFakeNode(n) for n in ("a1", "a2", "a3", "a100", "b3", "b5")
        )
        bufA = _PeakMemFakeBuffer("bufA", {b3}, 100, 100)
        bufB = _PeakMemFakeBuffer("bufB", {b5}, 200, 200)
        bufC = _PeakMemFakeBuffer("bufC", set(), 50, 50)
        bufD = _PeakMemFakeBuffer("bufD", set(), 999, 999)
        a1._outputs = [bufA]
        a2._outputs = [bufB]
        a3._outputs = [bufC]
        a100._outputs = [bufD]
        b3.mpi_node.pred_buffers = {bufA}
        b5.mpi_node.pred_buffers = {bufB}
        steps = {a1: 1, a2: 2, a3: 3, a100: 100, b3: 3, b5: 5}
        nodes_in_window = [a1, a2, a3, b3, b5]

        peak = mem_mod.estimate_region_peak_memory(
            nodes_in_window,
            region_start=0,
            region_end=5,
            step_of=lambda n: steps[n],
            graph_outputs={"bufC"},
        )
        # Walk per slot (alloc -> peak check -> free):
        #   slot 0: nothing                                 -> cur=0
        #   slot 1: a1 allocs bufA (+100)                   -> cur=100
        #   slot 2: a2 allocs bufB (+200)                   -> cur=300
        #   slot 3: a3 allocs bufC (+50), b3 frees bufA     -> peak=350, cur=250
        #   slot 4: nothing                                 -> cur=250
        #   slot 5: b5 frees bufB (-200)                    -> cur=50
        # a100 (step 100) is outside the window, so bufD is never seen.
        # bufC is a graph output, so it is never freed.
        self.assertEqual(peak, 350)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_GPU_AND_TRITON:
        run_tests(needs="filelock")
