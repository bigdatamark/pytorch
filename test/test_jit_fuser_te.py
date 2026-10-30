# Owner(s): ["NNC"]
# ruff: noqa: F841
# [MODULE DOC] Tests for the NNC / TensorExpr (TE) fuser in TorchScript. Defines helper
# context managers and graph-inspection utilities, the TestTEFuser suite (specialized into
# static and dynamic shape variants), an OpInfo driven suite that compiles ops with
# TensorExprKernel and compares against eager, and a loopnest randomization suite. Import
# time side effects: forces the profiling executor and graph executor optimization on before
# common_utils infers its executor setting, parses command line arguments when run as a
# script, deletes the base TestTEFuser class so only subclasses run, and instantiates
# device-generic test classes into the module globals. Running as a script calls the test
# runner unless Python is 3.14 or newer.

import contextlib
import math
import operator
import os
import sys
import unittest
import warnings

import torch
import torch.nn.functional as F
from torch.testing import FileCheck


# these needs to be set before `common_utils`
# infers `GRAPH_EXECUTOR`.
# this file **requires** these settings
# and setting them after `GRAPH_EXECUTOR` is
# inferred erroneously runs or skips
# some tests
torch._C._jit_set_profiling_executor(True)
torch._C._get_graph_executor_optimize(True)

if __name__ == "__main__":
    from torch.testing._internal.common_utils import parse_cmd_line_args

    # The value of GRAPH_EXECUTOR depends on command line arguments so make sure they're parsed
    # before instantiating tests.
    parse_cmd_line_args()

from itertools import combinations, permutations, product
from textwrap import dedent

from jit.test_fuser_common import TestFuserCommon  # noqa: F401
from test_jit import (
    backward_graph,
    get_lstm_inputs,
    get_milstm_inputs,
    LSTMCellC,
    LSTMCellF,
    LSTMCellS,
    MiLSTMCell,
)

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    enable_profiling_mode_for_profiling_tests,
    GRAPH_EXECUTOR,
    IS_FBCODE,
    IS_LINUX,
    ProfilingMode,
    run_tests,
    skipIfTorchDynamo,
    slowTest,
    TEST_WITH_ASAN,
)
from torch.testing._internal.jit_metaprogramming_utils import create_traced_fn
from torch.testing._internal.jit_utils import (
    clone_inputs,
    get_traced_sample_variant_pairs,
    JitTestCase,
    NoTracerWarnContextManager,
    RUN_CUDA,
    RUN_CUDA_HALF,
    RUN_CUDA_MULTI_GPU,
    set_fusion_group_inlining,
    TensorExprTestOptions,
    warmup_backward,
)


FUSION_GROUP = "prim::TensorExprGroup"
LLVM_ENABLED = torch._C._llvm_enabled()

autograd_check_set = {
    "aten::__is__",
    "prim::AutogradAllNonZero",
    "prim::AutogradAllZero",
    "prim::ListConstruct",
}


def strip_profiling_nodes(nodes):
    # [FUNCTION DOC] HELPER: Returns the nodes of the given sequence whose kind is not a
    # profiling bailout marker (BailoutTemplate or BailOut). The input is not modified; a
    # new list is returned.
    profiling_opcodes = {"prim::BailoutTemplate", "prim::BailOut"}
    return [n for n in nodes if n.kind() not in profiling_opcodes]


def warmup_forward(f, *args, profiling_count=2):
    # [FUNCTION DOC] HELPER: Calls the callable f with the given arguments profiling_count
    # times (default 2) so the profiling executor can record shapes and optimize. Returns
    # the result of the last call.
    for _ in range(profiling_count):
        results = f(*args)

    return results


@contextlib.contextmanager
def texpr_reductions_enabled():
    # [FUNCTION DOC] HELPER: Context manager that enables TensorExpr reduction fusion for
    # its body. Remembers the previous setting and restores it in a finally clause, even on
    # error.
    old = torch._C._jit_set_texpr_reductions_enabled(True)
    try:
        yield
    finally:
        torch._C._jit_set_texpr_reductions_enabled(old)


@contextlib.contextmanager
def texpr_enable_strategy(strategy):
    # [FUNCTION DOC] HELPER: Context manager that sets the JIT fusion strategy (a list of
    # kind/count pairs such as dynamic or static with a limit) for its body. Restores the
    # previous strategy in a finally clause.
    old = torch._C._jit_set_fusion_strategy(strategy)
    try:
        yield
    finally:
        torch._C._jit_set_fusion_strategy(old)


@contextlib.contextmanager
def inline_fusion_groups():
    # [FUNCTION DOC] HELPER: Context manager that turns on fusion group inlining (debug
    # flag) for its body. Restores the previous flag in a finally clause.
    old_inlining = torch._C._debug_get_fusion_group_inlining()
    torch._C._debug_set_fusion_group_inlining(True)
    try:
        yield
    finally:
        torch._C._debug_set_fusion_group_inlining(old_inlining)


class TestTEFuser(JitTestCase):
    # [CLASS DOC] Base suite of TE fuser tests, run only through its static and dynamic
    # shape subclasses (the base is deleted from the module). Subclasses must define the
    # dynamic_shapes attribute. setUp saves TensorExprTestOptions and the fusion strategy;
    # helpers check that a graph is fully fused and find fusion groups. Tests script or
    # trace small functions and assert fusion happens and results match eager.
    def setUp(self):
        # [FUNCTION DOC] Prepares each test: applies TensorExprTestOptions, sets the fusion
        # strategy to dynamic or static (limit 20) according to the dynamic_shapes class
        # attribute and remembers the old one, and builds the lists of devices (cpu plus
        # cuda if available) and of integer, floating point and combined dtypes.
        super().setUp()
        self.tensorexpr_options = TensorExprTestOptions()

        # note: `self.dynamic_shapes` instantiated in specialization of class
        # defined below

        fusion_strategy = [("DYNAMIC", 20)] if self.dynamic_shapes else [("STATIC", 20)]
        self.old_fusion_strategy = torch._C._jit_set_fusion_strategy(fusion_strategy)

        self.devices = ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]
        self.int_dtypes = [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]
        self.fp_dtypes = [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ]
        self.dtypes = self.int_dtypes + self.fp_dtypes

    def tearDown(self):
        # [FUNCTION DOC] Restores the TensorExpr test options and the saved fusion strategy,
        # then calls the parent tearDown.
        self.tensorexpr_options.restore()
        torch._C._jit_set_fusion_strategy(self.old_fusion_strategy)
        super().tearDown()

    def assertAllFused(self, graph, except_for=None):
        # [FUNCTION DOC] Asserts that every node of the top level block of the graph is a
        # constant, a single guard, an allowed node kind from except_for (default empty), or
        # an If node whose previous node is a guard. Guards are type checks, requires-grad
        # checks, dynamic guards or autodiff all-nonzero/all-zero checks. Fails on any other
        # node kind and fails unless exactly one guard was found.
        except_for = except_for if except_for is not None else set()
        # TODO - upstream
        guards = (
            "prim::TypeCheck",
            "prim::RequiresGradCheck",
            "prim::TensorExprDynamicGuard",
        )
        guard_found = False

        def autodiff_guard(node):
            # [FUNCTION DOC] HELPER: Returns True if the node is an all-reduction over a
            # list construct that has an autodiff all-nonzero or all-zero input; this is how
            # autodiff guards appear. Returns False otherwise.
            if node.kind() != "aten::all":
                return False
            inps = list(node.inputs())
            if len(inps) != 1 or inps[0].node().kind() != "prim::ListConstruct":
                return False
            li_inps = list(inps[0].node().inputs())
            for li_inp in li_inps:
                if li_inp.node().kind() in (
                    "prim::AutogradAllNonZero",
                    "prim::AutogradAllZero",
                ):
                    return True
            return False

        def is_guard(node):
            # [FUNCTION DOC] HELPER: Returns True if the node is one of the known guard
            # kinds or an autodiff guard.
            return node.kind() in guards or autodiff_guard(node)

        for node in graph.block().nodes():
            if node.kind() == "prim::Constant":
                continue
            if is_guard(node):
                self.assertFalse(guard_found)
                guard_found = True
                continue
            if node.kind() in except_for:
                continue
            if node.kind() == "prim::If":
                self.assertTrue(is_guard(node.prev()))
                continue
            self.assertTrue(False, "Found unexpected node:" + node.kind())

        self.assertTrue(guard_found)

    def assertLastGraphAllFused(self):
        # [FUNCTION DOC] Asserts that the most recently executed optimized graph is fully
        # fused, using assertAllFused with default exceptions.
        self.assertAllFused(torch.jit.last_executed_optimized_graph())

    def findFusionGroups(self, graph):
        # [FUNCTION DOC] Recursively walks a graph (or block) and returns a list of the
        # subgraphs of all TensorExprGroup nodes, descending into the blocks of other nodes
        # such as If nodes. Fusion groups are not searched inside.
        result = []
        for n in graph.nodes():
            if n.kind() == FUSION_GROUP:
                result.append(n.g("Subgraph"))
                continue
            for block in n.blocks():
                result += self.findFusionGroups(block)
        return result

    def test_typecheck(self):
        # [FUNCTION DOC] Checks a fused add-multiply kernel is created for size one input
        # and that a larger input afterwards triggers a type check failure and
        # recompilation, so the scripted result still equals eager.
        a = torch.ones(1)

        def fused_kernel(a, b):
            # [FUNCTION DOC] HELPER: Eager function adding two tensors and multiplying by
            # two.
            return (a + b) * 2.0

        scripted = self.checkScript(fused_kernel, (a, a))
        graph = scripted.graph_for(a, a)
        # double check we fused
        fusion_groups = self.findFusionGroups(graph)
        self.assertEqual(len(fusion_groups), 1)
        # we use a bigger tensor now (size 2)
        # if we won't trigger a recompilation
        # we will still create a tensor up to (size 1)
        # if the type check fails
        a = torch.ones(2)
        # shape changed if we don't trigger recompilation
        # we would compute the wrong result silently
        self.assertEqual(scripted(a, a), fused_kernel(a, a))

    def test_sum_simple(self):
        # [FUNCTION DOC] Checks that a squared-sum reduction script function fuses
        # completely when TE reductions are enabled.
        def func(x):
            # [FUNCTION DOC] HELPER: Squares the input and returns the sum.
            x2 = x * x
            return x2.sum()

        with texpr_reductions_enabled():
            a = torch.tensor(list(range(15)), dtype=torch.float, device="cpu")
            a = a.reshape(5, 3)
            scripted = self.checkScript(func, (a,))
            self.assertLastGraphAllFused()

    def test_nop(self):
        # [FUNCTION DOC] Placeholder test that does nothing.
        pass

    def test_sum_dim(self):
        # [FUNCTION DOC] Checks that sum over a positive dim and over a negative dim
        # followed by a multiply are fully fused with reductions enabled.
        def func(x):
            # [FUNCTION DOC] HELPER: Sums over dim zero then multiplies by two.
            return x.sum((0,)) * 2

        def func_neg(x):
            # [FUNCTION DOC] HELPER: Sums over dim minus two then multiplies by two.
            return x.sum((-2,)) * 2

        with texpr_reductions_enabled():
            a = torch.tensor(list(range(15)), dtype=torch.float, device="cpu")
            a = a.reshape(5, 3)
            scripted = self.checkScript(func, (a,))
            self.assertLastGraphAllFused()
            scripted = self.checkScript(func_neg, (a,))
            self.assertLastGraphAllFused()

    def test_sum_keepdim_cast(self):
        # [FUNCTION DOC] Checks that a sum with keepdim and a dtype cast to double followed
        # by a multiply fuses fully with reductions enabled.
        def func(x):
            # [FUNCTION DOC] HELPER: Sum over dim zero with keepdim and double dtype, times
            # two.
            return x.sum((0,), keepdim=True, dtype=torch.double) * 2

        with texpr_reductions_enabled():
            a = torch.tensor(list(range(15)), dtype=torch.float, device="cpu")
            a = a.reshape(5, 3)

            self.checkScript(func, (a,))
            self.assertLastGraphAllFused()

    def test_abs(self):
        # [FUNCTION DOC] On every device checks that absolute value times two is fully
        # fused.
        for device in self.devices:

            def func(x):
                # [FUNCTION DOC] HELPER: Returns absolute value of x times two.
                return x.abs() * 2

            a = torch.randn(5, device=device)
            scripted = self.checkScript(func, (a,))
            self.assertLastGraphAllFused()

    def test_unsqueeze_size_calculation(self):
        # [FUNCTION DOC] On every device checks that an unsqueeze followed by scale,
        # broadcast add and divide fuses fully when one input requires grad.
        for device in self.devices:

            def foo(b, d):
                # [FUNCTION DOC] HELPER: Unsqueezes d at dim one, multiplies by 42, adds b
                # and divides by 42.
                x = d.unsqueeze(1)
                y = x * 42.0
                z = b + y
                r = z / 42.0
                return r

            inputs = (
                torch.rand(20, 28, device=device, requires_grad=True),
                torch.rand(20, device=device),
            )
            scripted = self.checkScript(foo, inputs)
            self.assertAllFused(scripted.graph_for(*inputs))

    def test_zero_element_tensors(self):
        # [FUNCTION DOC] On every device checks that a script function on zero-element
        # tensors (atan2 of floats) runs and matches eager.
        for device in self.devices:

            def decode(sin_t, cos_t):
                # [FUNCTION DOC] HELPER: Returns atan2 of the sine and cosine tensors
                # converted to float.
                theta = torch.atan2(sin_t.float(), cos_t.float())
                return theta

            sin = torch.zeros(0, device=device)
            cos = torch.zeros(0, device=device)
            inputs = [sin, cos]
            ge = self.checkScript(decode, inputs)

    def test_arg_configurations_smoke(self):
        # [FUNCTION DOC] Smoke test that a traced function with a chunk gives the same
        # results for contiguous and non-contiguous (transposed) arguments, so one kernel is
        # not reused wrongly. Skipped for dynamic shapes.
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        # A smoke test to make sure we won't use the same kernel for contiguous
        # and non-contiguous arguments.
        # TODO: add optionally enabled debug counters to the fuser to verify
        #       that we really can tell the difference between configurations
        for device in self.devices:

            def f(x, y):
                # [FUNCTION DOC] HELPER: Adds two tensors, chunks into two along dim one and
                # multiplies the halves.
                z1, z2 = (x + y).chunk(2, dim=1)
                return z1 * z2

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            traced_f = torch.jit.trace(f, (x, y))
            self.assertEqual(traced_f(x.t().contiguous(), y), traced_f(x.t(), y))

    def test_broadcast(self):
        # [FUNCTION DOC] On every device checks that scale-and-shift with broadcast operands
        # of different rank is scripted correctly.
        for device in self.devices:

            def scaleshift(x, scale, shift):
                # [FUNCTION DOC] HELPER: Returns x times scale plus shift.
                return x * scale + shift

            inputs = [
                torch.randn(4, 4, dtype=torch.float, device=device),
                torch.randn(4, dtype=torch.float, device=device),
                torch.randn(4, dtype=torch.float, device=device),
            ]
            self.checkScript(scaleshift, inputs)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_HALF, "no half support")
    @unittest.skipIf(
        GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no half support with profiling on"
    )
    def test_cuda_half(self):
        # [FUNCTION DOC] CUDA-only test (skipped without CUDA, without half support, or
        # unless the legacy executor is used) that compares half precision traced fusion
        # outputs and gradients against float32 eager for comparison, relu and exp
        # functions.
        x = torch.randn(4, 4, dtype=torch.half, device="cuda")
        y = torch.randn(4, 4, dtype=torch.half, device="cuda")

        funcs = [self.fn_test_comparison_gt_lt, self.fn_test_relu, self.fn_test_exp]

        # Note: Non fused inputs must be float to prevent loss of precision
        inputs = (x.float(), y.float())
        fusion_inputs = (x, y)
        for fn in funcs:
            local_inputs = [t.clone().requires_grad_() for t in inputs]
            local_fusion_inputs = [t.clone().requires_grad_() for t in fusion_inputs]

            # Verifies outputs
            fusion = torch.jit.trace(fn, local_fusion_inputs, check_trace=False)
            outputs = fn(*local_inputs)
            fusion_outputs = fusion(*local_fusion_inputs)
            outputs_half = [t.half() for t in outputs]
            self.assertEqual(outputs_half, fusion_outputs)

            # Verifies gradients
            for output, fusion_output in zip(outputs_half, fusion_outputs):
                grads = torch.autograd.grad(
                    output.float().sum(),
                    local_inputs,
                    allow_unused=True,
                    retain_graph=True,
                )
                fusion_grads = torch.autograd.grad(
                    fusion_output.sum(),
                    local_fusion_inputs,
                    allow_unused=True,
                    retain_graph=True,
                )
                grads_half = [t.half() for t in grads]
                self.assertEqual(grads_half, fusion_grads)

    def test_checks_cat_inputs(self):
        # [FUNCTION DOC] With fusion group inlining enabled, checks on each device that a
        # concatenation of two fused expressions with a broadcastable second input yields
        # output shape three by four (inputs are not treated as broadcasting) and is fully
        # fused.
        # single fusion node causes error
        with set_fusion_group_inlining(True):
            for device in self.devices:
                # We shouldn't treat cat nodes as broadcasting. All their inputs
                # need to be checked for having the same map size, before we can
                # run the kernel.
                def f(x, y):
                    # [FUNCTION DOC] HELPER: Concatenates two separately computed polynomial
                    # expressions of x and y along dim zero.
                    return torch.cat([x + 2 * x + x**2, y + 4 * y + y**3], dim=0)

                # NOTE: y is broadcastable to x, but output of f(x, y) should have
                # shape 3x4, and not 4x4.
                x = torch.randn(2, 4, dtype=torch.float, device=device)
                y = torch.randn(1, 4, dtype=torch.float, device=device)

                scripted = self.checkScript(f, (x, y))
                self.assertEqual(scripted(x, y).shape, (3, 4))
                self.assertAllFused(scripted.graph_for(x, y))

    def test_chunk(self):
        # [FUNCTION DOC] On each device checks a three-way chunk along dim one followed by
        # arithmetic is scripted correctly and fully fused. Skipped for dynamic shapes.
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:

            def fn(x):
                # [FUNCTION DOC] HELPER: Chunks x in three along dim one and returns a times
                # b plus c.
                a, b, c = x.chunk(3, 1)
                return a * b + c

            inputs = [torch.randn(10, 6, dtype=torch.float, device=device)]

            self.checkScript(fn, inputs)
            self.assertLastGraphAllFused()

    def test_chunk_correctness(self):
        # [FUNCTION DOC] Checks chunk results and full fusion for four-way chunks along the
        # first, middle and last dim over a split-size-one tensor, a contiguous tensor and a
        # transposed tensor. Skipped for dynamic shapes.
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:

            def chunk_4_0(x):
                # [FUNCTION DOC] HELPER: Sums the four chunks of x along dim zero.
                x0, x1, x2, x3 = x.chunk(4, 0)
                return x0 + x1 + x2 + x3

            def chunk_4_1(x):
                # [FUNCTION DOC] HELPER: Sums the four chunks of x along dim one.
                x0, x1, x2, x3 = x.chunk(4, 1)
                return x0 + x1 + x2 + x3

            def chunk_4_last(x):
                # [FUNCTION DOC] HELPER: Sums the four chunks of x along dim two.
                x0, x1, x2, x3 = x.chunk(4, 2)
                return x0 + x1 + x2 + x3

            fns = [chunk_4_0, chunk_4_1, chunk_4_last]
            tensors = [
                # splitSize = 1
                torch.randn(4, 4, 4, dtype=torch.float, device=device),
                # contiguous case
                torch.randn(12, 8, 16, dtype=torch.float, device=device),
                # non-contiguous case
                torch.randn(12, 8, 16, dtype=torch.float, device=device).transpose(
                    1, 2
                ),
            ]

            for tensor in tensors:
                for fn in fns:
                    self.checkScript(fn, [tensor])
                    self.assertLastGraphAllFused()

    def test_chunk_distributes(self):
        # [FUNCTION DOC] Traces a function that chunks a sum and multiplies the halves, then
        # checks the printed fusion graph has exactly one ConstantChunk inside a fusion
        # group. Skipped for dynamic shapes (the duplicate skip check is redundant).
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:

            def f(x, y):
                # [FUNCTION DOC] HELPER: Adds x and y, chunks in two along dim one and
                # multiplies the halves.
                z1, z2 = (x + y).chunk(2, dim=1)
                return z1 * z2

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(f, (x, y))
            graph = ge.graph_for(x, y)
            # XXX: The old fuser does broadcast_tensors but the new fuser doesn't.
            # FileCheck().check("broadcast_tensors").check('with ' + FUSION_GROUP + '_') \
            #     .check_count('ConstantChunk', 2, exactly=True).run(str(graph))
            FileCheck().check("with " + FUSION_GROUP + "_").check_count(
                "ConstantChunk", 1, exactly=True
            ).run(str(graph))

    def test_chunk_motion_deduplicates_inputs(self):
        # [FUNCTION DOC] Checks that chunks of an intermediate tensor, used with duplicate
        # inputs, still fuse fully for two functions. Skipped for dynamic shapes.
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:

            def func1(x):
                # [FUNCTION DOC] HELPER: Squares x, chunks in two and multiplies the halves.
                z = x * x
                z0, z1 = z.chunk(2)
                return z0 * z1

            def func2(x):
                # [FUNCTION DOC] HELPER: Cubes x, chunks in two and multiplies the halves.
                z = x * x * x
                z0, z1 = z.chunk(2)
                return z0 * z1

            inputs = [torch.tensor([1.1, 1.2], device=device, dtype=torch.float)]
            for func in [func1, func2]:
                self.checkScript(func, inputs)
                self.assertLastGraphAllFused()

    def test_chunk_multiple(self):
        # [FUNCTION DOC] Checks fusion with several chunked inputs whose arguments are
        # intentionally out of order, with different chunk counts and dims. Skipped for
        # dynamic shapes.
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:
            # The arguments are intentionally used out of order as a test to see
            # if the fusion compiler adds extra args in the correct order
            def fn(s, x, y, z):
                # [FUNCTION DOC] HELPER: Chunks z, x and y along different dims and sums all
                # pieces with s.
                z1, z2 = z.chunk(2, 2)
                x1, x2, x3 = x.chunk(3, 1)
                y1, y2 = y.chunk(2, 0)
                return s + x1 + x2 + x3 + y1 + y2 + z1 + z2

            inputs = [
                torch.randn(5, 2, 3, dtype=torch.float, device=device),
                torch.randn(5, 6, 3, dtype=torch.float, device=device),
                torch.randn(10, 2, 3, dtype=torch.float, device=device),
                torch.randn(5, 2, 6, dtype=torch.float, device=device),
            ]

            ge = self.checkScript(fn, inputs)
            self.assertAllFused(ge.graph_for(*inputs))

    def test_minmax(self):
        # [FUNCTION DOC] Checks that elementwise max and min of two tensors, including with
        # a NaN scalar tensor operand, are fully fused on each device.
        for device in self.devices:

            def tmax(a, b):
                # [FUNCTION DOC] HELPER: Returns the maximum of twice a and b.
                return torch.max(2 * a, b)

            def tmin(a, b):
                # [FUNCTION DOC] HELPER: Returns the minimum of twice a and b.
                return torch.min(2 * a, b)

            a = torch.randn(4, 4, dtype=torch.float)
            b = torch.randn(4, 4, dtype=torch.float)
            nan = torch.tensor(float("nan"), dtype=torch.float)

            for f, inputs, device in product(
                (tmax, tmin), ([a, b], [a, nan], [b, nan]), self.devices
            ):
                inputs = [t.to(device) for t in inputs]
                s = self.checkScript(f, inputs)
                self.assertAllFused(s.graph_for(*inputs))

    def test_clamp(self):
        # [FUNCTION DOC] Checks clamp variants (finite bounds, infinite bounds, optional min
        # or max) with normal and NaN inputs are fused in forward and in the backward graph,
        # using profiling mode. The first input requires grad.
        for device in self.devices:

            def func2(a, b):
                # [FUNCTION DOC] HELPER: Clamps the sum to the range zero to two.
                return torch.clamp(a + b, min=0, max=2)

            def funcInf(a, b):
                # [FUNCTION DOC] HELPER: Clamps the sum from zero to positive infinity.
                return torch.clamp(a + b, min=0, max=float("inf"))

            def funcNegInf(a, b):
                # [FUNCTION DOC] HELPER: Clamps the sum from negative infinity to zero.
                return torch.clamp(a + b, min=float("-inf"), max=0)

            def funcOptMin(a, b):
                # [FUNCTION DOC] HELPER: Clamps the sum with only an upper bound of two.
                return torch.clamp(a + b, max=2)

            def funcOptMax(a, b):
                # [FUNCTION DOC] HELPER: Clamps the sum with only a lower bound of zero.
                return torch.clamp(a + b, min=0)

            a = torch.randn(4, 4, dtype=torch.float, device=device, requires_grad=True)
            b = torch.randn(4, 4, dtype=torch.float, device=device)
            nan = torch.tensor(float("nan"), dtype=torch.float, device=device)

            funcs = (func2, funcInf, funcNegInf, funcOptMin, funcOptMax)
            for f, inputs in product(funcs, [[a, b], [a, nan]]):
                inp1, inp2 = inputs
                s = self.checkScript(f, (inp1, inp2), profiling=ProfilingMode.PROFILING)
                self.assertAllFused(
                    s.graph_for(inp1, inp2),
                    except_for={"aten::size", "aten::_size_if_not_equal"},
                )
                c = s(inp1, inp2)
                with enable_profiling_mode_for_profiling_tests():
                    warmup_backward(c.sum())
                graph = backward_graph(s)
                self.assertAllFused(
                    graph,
                    except_for={"aten::Float", "aten::_grad_sum_to_size"}.union(
                        autograd_check_set
                    ),
                )

    def test_clamp_double(self):
        # [FUNCTION DOC] Checks a double-precision clamp with float scalar bounds is fused
        # (except the final subtraction) with tight tolerance.
        for device in self.devices:

            def clamp_double(x, eta: float):
                # [FUNCTION DOC] HELPER: Returns one minus x clamped to eta and one minus
                # eta.
                return 1 - x.clamp(eta, 1 - eta)

            x = torch.tensor([1.0, 1.0], dtype=torch.double, device=device)
            eta = 1e-9
            s = self.checkScript(
                clamp_double,
                (x, eta),
                profiling=ProfilingMode.PROFILING,
                atol=1e-10,
                rtol=1e-5,
            )
            self.assertAllFused(s.graph_for(x, eta), except_for={"aten::sub"})

    def test_clamp_int(self):
        # [FUNCTION DOC] Checks clamp of an integer tensor with a very large integer bound
        # (above 32 bits) is fused.
        for device in self.devices:

            def clamp_int(x, eta: int):
                # [FUNCTION DOC] HELPER: Clamps x between zero and the integer eta.
                return x.clamp(0, eta)

            x = torch.tensor([1, 1], device=device)
            eta = 1 << 32
            s = self.checkScript(clamp_int, (x, eta), profiling=ProfilingMode.PROFILING)
            self.assertAllFused(s.graph_for(x, eta))

    def test_add_bool(self):
        # [FUNCTION DOC] Traces boolean tensor addition of three operands over several sizes
        # and devices and checks full fusion.
        sizes = [(1,), (2,), (4, 4)]
        for device, size in product(self.devices, sizes):

            def f(x, y, z):
                # [FUNCTION DOC] HELPER: Adds three tensors.
                return x + y + z

            x = torch.randint(0, 2, size, dtype=torch.bool, device=device)
            y = torch.randint(0, 2, size, dtype=torch.bool, device=device)
            z = torch.randint(0, 2, size, dtype=torch.bool, device=device)
            ge = self.checkTrace(f, (x, y, z), inputs_require_grads=False)
            self.assertAllFused(ge.graph_for(x, y, z))

    def test_mul_bool(self):
        # [FUNCTION DOC] Traces boolean tensor multiplication of three operands on each
        # device and checks full fusion.
        for device in self.devices:

            def f(x, y, z):
                # [FUNCTION DOC] HELPER: Multiplies three tensors.
                return x * y * z

            x = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            y = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            z = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)

            ge = self.checkTrace(f, (x, y, z), inputs_require_grads=False)
            self.assertAllFused(ge.graph_for(x, y, z))

    def test_div_bool(self):
        # [FUNCTION DOC] Traces division of a boolean sum by an all-true boolean tensor on
        # each device and checks full fusion.
        for device in self.devices:

            def f(x, y, z):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y divided by z.
                return (x + y) / z

            x = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            y = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            z = torch.ones_like(x, dtype=torch.bool, device=device)

            ge = self.checkTrace(f, (x, y, z), inputs_require_grads=False)
            self.assertAllFused(ge.graph_for(x, y, z))

    def test_bitwise_ops(self):
        # [FUNCTION DOC] For every integer dtype, bitwise operator and device, builds data,
        # skips combinations eager rejects, traces a two-step application and checks
        # equality and full fusion; failures are re-raised as a RuntimeError naming dtype,
        # operator and device.
        def apply(fn):
            # [FUNCTION DOC] HELPER: Returns a function applying a binary function twice on
            # three inputs.
            return lambda x, y, z: fn(fn(x, y), z)

        binary_ops = [
            operator.__and__,
            operator.__or__,
            operator.__xor__,
            operator.__lshift__,
            operator.__rshift__,
        ]
        devices = self.devices
        for dtype, op, device in product(self.int_dtypes, binary_ops, devices):
            try:
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                z = self.data_for(dtype, device)
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (x, y, z))
                self.assertEqual(ref, t(x, y, z))
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    def test_minmax_int_ops(self):
        # [FUNCTION DOC] Like the bitwise test but for integer min and max over all integer
        # dtypes and devices: compares traced results with eager and checks full fusion,
        # raising RuntimeError with details on failure.
        def apply(fn):
            # [FUNCTION DOC] HELPER: Returns a function applying a binary function twice on
            # three inputs.
            return lambda x, y, z: fn(fn(x, y), z)

        binary_ops = [torch.min, torch.max]
        devices = self.devices
        for dtype, op, device in product(self.int_dtypes, binary_ops, devices):
            try:
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                z = self.data_for(dtype, device)
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (x, y, z))
                self.assertEqual(ref, t(x, y, z))
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    def test_comparison_eq_ne(self):
        # [FUNCTION DOC] Traces equality and inequality masks used to blend tensors on each
        # device and checks full fusion.
        for device in self.devices:

            def f(x, y):
                # [FUNCTION DOC] HELPER: Builds masks from x equal to zero and not equal to
                # zero and blends them with y.
                mask = (x == 0).type_as(x)
                z = x * mask + y
                mask = (x != 0).type_as(x)
                z = z * mask + y
                return z

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(f, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    @staticmethod
    def fn_test_comparison_gt_lt(x, y):
        # [FUNCTION DOC] HELPER: Static function shared by tests; blends x and y using
        # greater-than and less-than masks of x and returns the result.
        mask = (x > 0).type_as(x)
        z = x * mask + y
        mask = (x < 0).type_as(x)
        z = z * mask + y
        return z

    def test_comparison_gt_lt(self):
        # [FUNCTION DOC] Traces the shared greater-than/less-than function on each device
        # and checks full fusion.
        for device in self.devices:
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(self.fn_test_comparison_gt_lt, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    def test_comparison_ge_le(self):
        # [FUNCTION DOC] Traces a greater-or-equal / less-or-equal blending function on each
        # device, checks full fusion, then marks inputs as requiring grad and checks fusion
        # again allowing size related nodes.
        for device in self.devices:

            def f(x, y):
                # [FUNCTION DOC] HELPER: Blends x and y using masks of x at least zero and
                # at most zero.
                mask = (x >= 0).type_as(x)
                z = x * mask + y
                mask = (x <= 0).type_as(x)
                z = z * mask + y
                return z

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(f, (x, y))
            self.assertAllFused(ge.graph_for(x, y))
            x.requires_grad_(True)
            y.requires_grad_(True)
            self.assertAllFused(
                ge.graph_for(x, y),
                except_for=(
                    "aten::size",
                    "prim::BroadcastSizes",
                    "aten::_size_if_not_equal",
                ),
            )

    def test_addcmul(self):
        # [FUNCTION DOC] Checks a traced addcmul with broadcast inputs yields exactly one
        # fusion group containing an add and an addcmul.
        for device in self.devices:
            t = torch.randn(1, 4, dtype=torch.float, device=device)
            t1 = torch.randn(4, 1, dtype=torch.float, device=device)
            t2 = torch.randn(1, 4, dtype=torch.float, device=device)

            def foo(t, t1, t2):
                # [FUNCTION DOC] HELPER: Calls addcmul on t with t plus one and t2, value
                # 0.1.
                return t.addcmul(t + 1, t2, value=0.1)

            ge = self.checkTrace(foo, (t, t1, t2), allow_unused=True)
            graph = ge.graph_for(t, t1, t2)
            fusion_groups = self.findFusionGroups(graph)
            self.assertEqual(len(fusion_groups), 1)
            FileCheck().check("aten::add(").check("aten::addcmul(").run(
                str(fusion_groups[0])
            )

    # TODO: We leak CUDA memory here because the traced graph holds onto a
    # constant-ified tensor. Since the Python-global CompilationUnit is alive
    # until the end of the process, the memory is effectively leaked.
    # Removed `_cuda` suffix from this test which disables leak-checking.
    # If this is a real problem, we'll need to revisit Torchscript Function
    # lifetimes in Python.
    def test_lerp(self):
        # [FUNCTION DOC] On each device checks lerp with a scalar weight is traced and fully
        # fused. The tensor weight variant is disabled.
        for device in self.devices:
            start = torch.randn(4, 1, dtype=torch.float, device=device)
            end = torch.randn(1, 4, dtype=torch.float, device=device)
            weight = torch.tensor(0.5, dtype=torch.float, device=device)

            # scalar weight overload
            def foo_weight_scalar(start, end):
                # [FUNCTION DOC] HELPER: Lerp of start plus one toward end with scalar
                # weight 0.5.
                return torch.lerp(start + 1, end, 0.5)

            # tensor weight overload
            def foo_weight_tensor(start, end):
                # [FUNCTION DOC] HELPER: Lerp of start plus one toward end with the captured
                # tensor weight; defined but not used because the check is commented out.
                return torch.lerp(start + 1, end, weight)

            ge_weight_scalar = self.checkTrace(foo_weight_scalar, (start, end))
            graph = ge_weight_scalar.graph_for(start, end)
            self.assertAllFused(graph)

            # TODO: uncomment when TE enables support for scalar tensors
            # ge_weight_tensor = self.checkTrace(foo_weight_tensor, (start, end))
            # graph = ge_weight_tensor.graph_for(start, end)
            # self.assertAllFused(graph)

    def test_concat(self):
        # [FUNCTION DOC] With inlining of fusion groups enabled, checks a concatenation of a
        # sum and a product is fully fused on each device.
        # disabling concat causes error with single concat node
        with set_fusion_group_inlining(True):
            for device in self.devices:
                hx = torch.randn(3, 20, dtype=torch.float, device=device)
                cx = torch.randn(3, 20, dtype=torch.float, device=device)

                def foo(hx, cx):
                    # [FUNCTION DOC] HELPER: Concatenates the sum and the product of hx and
                    # cx.
                    return torch.cat((hx + cx, hx * cx))

                ge = self.checkTrace(foo, (hx, cx))
                graph = ge.graph_for(hx, cx)
                self.assertAllFused(graph)
                # XXX: TE fuser can handle concats in a fusion group.
                # FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    def test_remove_output_used_only_in_size(self):
        # [FUNCTION DOC] Scripts a function, warms it up three times, then checks the
        # differentiable graph subgraph has one If node whose outputs (and so the fusion
        # group inside) number one, meaning an output used only for sizes was removed.
        for device in self.devices:

            def test_fuse(a, b):
                # [FUNCTION DOC] HELPER: Returns the add of a, b and b again.
                c = a + b
                d = c + b
                return d

            scripted_f = torch.jit.script(test_fuse)
            x = torch.ones(1, requires_grad=True, device=device)
            y = torch.ones(1, requires_grad=True, device=device)
            warmup_forward(scripted_f, x, y, profiling_count=3)
            g = scripted_f.graph_for(x, y)
            diff_nodes = g.findAllNodes("prim::DifferentiableGraph")
            self.assertEqual(len(diff_nodes), 1)
            g = diff_nodes[0].g("Subgraph")
            if_nodes = [n for n in g.nodes() if n.kind() == "prim::If"]
            self.assertEqual(len(if_nodes), 1)

            # the if node and the fusion group inside it should only have one output
            self.assertEqual(len(list(if_nodes[0].outputs())), 1)

    def test_concat_invariant(self):
        # [FUNCTION DOC] Checks concat followed by an add is fused except an add, covering
        # the invariant that a concat output is not consumed inside its own group.
        for device in self.devices:
            # Invariant: the output of prim::FusedConcat may
            # not be an input to any node inside the FusionGroup.
            def fn(x, y, z):
                # [FUNCTION DOC] HELPER: Concatenates x plus y with x minus y and adds z.
                x1 = x + y
                y1 = x - y
                w = torch.cat([x1, y1])
                return w + z

            x = torch.randn(2, 2, dtype=torch.float, device=device)
            y = torch.randn(2, 2, dtype=torch.float, device=device)
            z = torch.randn(4, 2, dtype=torch.float, device=device)
            ge = self.checkTrace(fn, (x, y, z))
            graph = ge.graph_for(x, y, z)
            self.assertAllFused(graph, except_for={"aten::add"})
            # XXX: TE fuser can handle concats inside a fusion group.
            # FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    @staticmethod
    def fn_test_exp(x, y):
        # [FUNCTION DOC] HELPER: Static function shared by tests returning the exponential
        # of x plus half of y.
        return (x + 0.5 * y).exp()

    def test_exp(self):
        # [FUNCTION DOC] Traces the shared exp function on each device and checks full
        # fusion.
        for device in self.devices:
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(self.fn_test_exp, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    def test_threshold(self):
        # [FUNCTION DOC] Scripts a threshold plus additions function on each device and
        # checks it is fully fused.
        for device in self.devices:

            def f(x):
                # [FUNCTION DOC] HELPER: Thresholds x at zero with value minus ten and adds
                # x three times.
                return torch.threshold(x, 0, -10) + x + x + x

            x = torch.tensor([-1, -0.5, 0, 1, 2, 3], device=device)
            scripted = self.checkScript(f, (x,))
            self.assertAllFused(scripted.graph_for(x))

    def test_scalar_arg(self):
        # [FUNCTION DOC] Checks scripted functions with a float scalar argument (an integer
        # is passed) are fused, both without and with a requires-grad input (using a second
        # function to avoid bailouts).
        for device in self.devices:

            def fn_test_scalar_arg(x: torch.Tensor, p: float) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns p times the sum of x squared and x.
                return p * (x * x + x)

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            p = 3
            scripted = self.checkScript(fn_test_scalar_arg, (x, p))
            self.assertAllFused(scripted.graph_for(x, p))

            x.requires_grad_(True)

            # use another function otherwise we will bailout
            # and won't be able to do fused checks
            def fn_test_scalar_arg_requires_grad(
                x: torch.Tensor, p: float
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Same computation as the scalar-argument function,
                # defined separately so the requires-grad case does not reuse a bailed-out
                # profile.
                return p * (x * x + x)

            scripted = torch.jit.script(fn_test_scalar_arg_requires_grad)
            out = scripted(x, p)
            out = scripted(x, p)
            out = scripted(x, p)
            self.assertAllFused(
                scripted.graph_for(x, p),
                except_for=(
                    "aten::size",
                    "prim::BroadcastSizes",
                    "aten::_size_if_not_equal",
                ),
            )

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    def test_fusion_reuse_multi_gpu(self):
        # [FUNCTION DOC] CUDA multi-GPU test checking that a scripted function first run on
        # CPU and then on two different cuda devices does not crash and uses different
        # kernels.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x times y times x times y.
            return x * y * x * y

        inputs_cpu = [
            torch.randn(4, 4, dtype=torch.float),
            torch.randn(4, 4, dtype=torch.float),
        ]
        inputs_cuda0 = [x.cuda(0) for x in inputs_cpu]
        inputs_cuda1 = [y.cuda(1) for y in inputs_cpu]

        # Should not crash; these should compile different kernels.
        ge = self.checkScript(fn, inputs_cpu)
        self.assertAllFused(ge.graph_for(*inputs_cpu))
        ge(*inputs_cuda0)
        ge(*inputs_cuda1)

    # TODO: we're currently not checking 'device' in the type info when pulling
    # nodes into a fusion group. We should fix that and re-enable this test.
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    def test_kernel_cache_multi_gpu(self):
        # [FUNCTION DOC] CUDA multi-GPU test that three identical fusion groups on cpu and
        # two cuda devices exist; the kernel cache size assertion is commented out. A TODO
        # notes device is not checked when forming groups.
        def not_fusible(x):
            # [FUNCTION DOC] HELPER: Identity function that blocks fusion between outputs.
            return x

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Computes fifth powers by repeated multiplication of
            # three inputs and returns them through the identity helper.
            x_out = x * x * x * x * x  # fusion: lambda x. x * x * x * x * x
            y_out = y * y * y * y * y
            z_out = z * z * z * z * z
            return not_fusible(x_out), not_fusible(y_out), not_fusible(z_out)

        inputs = [
            torch.randn(4, 4, dtype=torch.float),
            torch.randn(4, 4, dtype=torch.float, device="cuda:0"),
            torch.randn(4, 4, dtype=torch.float, device="cuda:1"),
        ]

        prev_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()

        # There are 3 FusionGroups. Because they have the same graph, they
        # should reuse the same KernelSpec in the KernelSpec cache.
        ge = self.checkScript(fn, inputs)
        self.assertGraphContainsExactly(ge.graph_for(*inputs), FUSION_GROUP, 3, True)
        new_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()
        # XXX: This assumes that the same kernel isn't already used by another test
        # FIXME: Use the TE fuser's way of querying the cache.
        # self.assertEqual(new_cache_size - prev_cache_size, 1)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    def test_nonzero_device_cuda(self):
        # [FUNCTION DOC] Checks fusion on cuda device index one for a traced sigmoid-tanh
        # function. Skipped without multiple GPUs.
        device = "cuda:" + str(1)
        x = torch.tensor([0.4], dtype=torch.float, device=device)
        y = torch.tensor([0.7], dtype=torch.float, device=device)

        def doit(x, y):
            # [FUNCTION DOC] HELPER: Sigmoid of tanh of x times the sum of x and y plus x.
            return torch.sigmoid(torch.tanh(x * (x + y) + x))

        ge = self.checkTrace(doit, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    def test_lstm(self):
        # [FUNCTION DOC] On each device scripts the scripted LSTM cell with training inputs
        # and checks it is fully fused except tuple construction.
        for device in self.devices:
            inputs = get_lstm_inputs(device, training=True)
            module = self.checkScript(LSTMCellS, inputs)
            self.assertAllFused(
                module.graph_for(inputs), except_for={"prim::TupleConstruct"}
            )

    def test_lstm_concat(self):
        # [FUNCTION DOC] With inlining enabled, traces the LSTM cell with concat on each
        # device and checks full fusion allowing tuple construction and linear, and extra
        # nodes under dynamic shapes.
        # single fusion node causes error
        with set_fusion_group_inlining(True):
            for device in self.devices:
                inputs = get_lstm_inputs(device)
                ge = self.checkTrace(LSTMCellC, inputs)
                graph = ge.graph_for(*inputs)
                except_nodes = {"prim::TupleConstruct", "aten::linear"}
                # TODO... Chunk
                if self.dynamic_shapes:
                    except_nodes = except_nodes.union(
                        {"aten::add", "prim::ConstantChunk"}
                    )
                self.assertAllFused(ge.graph_for(*inputs), except_for=except_nodes)
                # XXX: TE fuser can handle concats inside a fusion group.
                # FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    def test_lstm_gates_permutations(self):
        # [FUNCTION DOC] For each permutation of the four LSTM gate terms builds source for
        # a cell function, evaluates it in Python and as a TorchScript compilation unit,
        # compares outputs and checks the forward graph has the expected number of fusion
        # groups (two for dynamic shapes, else one).
        for device in self.devices:
            # lstm has gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh.
            # Test that any permutation of this will still result in one FusionGroup.
            choices = ["x.mm(w_ih.t())", "hx.mm(w_hh.t())", "b_ih", "b_hh"]
            template = dedent(
                """
            def cell(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
                gates = {} + {} + {} + {}
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                return ingate * forgetgate * cellgate * outgate
            """
            )
            for permutation in permutations(choices, len(choices)):
                code = template.format(*permutation)
                scope = {}
                exec(code, globals(), scope)
                cu = torch.jit.CompilationUnit(code)
                fusion_group_len = 2 if self.dynamic_shapes else 1
                inputs = get_lstm_inputs(device, training=False)
                self.assertEqual(cu.cell(*inputs), scope["cell"](*inputs))
                forward_graph = cu.cell.graph_for(*inputs)
                self.assertGraphContainsExactly(
                    forward_graph, FUSION_GROUP, fusion_group_len
                )

    # TODO: Fuser doesn't work at all when inputs require grad. Fix that
    def test_lstm_traced(self):
        # [FUNCTION DOC] Traces the LSTM cell on each device and checks the expected count
        # of fusion groups and that the relevant group contains sigmoid and tanh (and chunk
        # when static).
        for device in self.devices:
            inputs = get_lstm_inputs(device)
            ge = self.checkTrace(LSTMCellF, inputs)
            graph = ge.graph_for(*inputs)
            fusion_groups = self.findFusionGroups(graph)
            # TODO: chunk
            fusion_group_len = 2 if self.dynamic_shapes else 1
            self.assertEqual(len(fusion_groups), fusion_group_len)
            f = FileCheck()
            if not self.dynamic_shapes:
                f.check("Chunk")
            f.check("aten::sigmoid").check("aten::tanh").run(
                str(fusion_groups[0 if not self.dynamic_shapes else 1])
            )

    def test_milstm(self):
        # [FUNCTION DOC] Scripts the multiplicative LSTM cell, checks the fusion group
        # count, graph structure and runs a backward pass. Skipped for dynamic shapes.
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        for device in self.devices:
            inputs = get_milstm_inputs(device, training=True)
            module = self.checkScript(MiLSTMCell, inputs)
            forward_graph = module.graph_for(*inputs)
            # TODO: chunk
            fusion_group_len = 2 if self.dynamic_shapes else 1
            self.assertGraphContainsExactly(
                forward_graph, FUSION_GROUP, fusion_group_len, consider_subgraphs=True
            )
            FileCheck().check("DifferentiableGraph").check("TupleConstruct").check_next(
                "return"
            ).check(FUSION_GROUP).run(str(forward_graph))
            hy, cy = module(*inputs)
            warmup_backward((hy + cy).sum())

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skip("rand_like is not supported yet")
    def test_rand_cuda(self):
        # [FUNCTION DOC] Skipped (rand_like unsupported) CUDA test intended to show random
        # output differs between calls, lies in the range zero to one and is fused.
        class M(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module with a constant cuda device attribute used
            # in a skipped test.
            __constants__ = ["d"]

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent initializer and stores a cuda
                # device attribute.
                super().__init__()
                self.d = torch.device("cuda")

            @torch.jit.script_method
            def create(self, x):
                # [FUNCTION DOC] HELPER: Script method returning x squared plus x plus
                # random numbers shaped like x.
                return x * x + x + torch.rand_like(x)

        x = torch.zeros([3, 4, 5], dtype=torch.float, device="cuda")
        m = M()
        out1 = m.create(x)
        out2 = m.create(x)
        self.assertNotEqual(out1, out2)
        self.assertTrue(torch.all(out1 >= 0))
        self.assertTrue(torch.all(out1 < 1))
        self.assertTrue(torch.all(out2 >= 0))
        self.assertTrue(torch.all(out2 < 1))
        self.assertAllFused(m.create.graph_for(x))

    @staticmethod
    def fn_test_relu(x, y):
        # [FUNCTION DOC] HELPER: Static function shared by tests returning relu of x plus
        # half of y.
        return F.relu(x + 0.5 * y)

    def test_relu(self):
        # [FUNCTION DOC] Traces the shared relu function on each device and checks full
        # fusion.
        for device in self.devices:
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(self.fn_test_relu, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    def test_erf(self):
        # [FUNCTION DOC] On cuda devices only (cpu skipped) checks erf minus erfc then relu
        # is fused with profiling, and again with a requires-grad input allowing size
        # related nodes.
        for device in self.devices:
            # only enabled on gpu
            if device == "cpu":
                continue

            def fn_test_erf(x):
                # [FUNCTION DOC] HELPER: Relu of erf minus erfc of x.
                return F.relu(torch.erf(x) - torch.erfc(x))

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            ge = self.checkScript(fn_test_erf, (x,), profiling=ProfilingMode.PROFILING)
            self.assertAllFused(ge.graph_for(x))
            x.requires_grad_(True)
            ge = self.checkScript(fn_test_erf, (x,), profiling=ProfilingMode.PROFILING)
            self.assertAllFused(
                ge.graph_for(x),
                except_for=(
                    "aten::size",
                    "prim::BroadcastSizes",
                    "aten::_size_if_not_equal",
                ),
            )

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skip("rand_like is not supported yet")
    def test_rand_broadcast_cuda(self):
        # [FUNCTION DOC] Skipped CUDA test for random numbers broadcast in a fused script
        # function, checking fusion and that broadcasted random values are consistent across
        # rows.
        def fn_test_rand(x, y):
            # [FUNCTION DOC] HELPER: Multiplies a random tensor shaped like y by x and adds
            # x.
            r = torch.rand_like(y)
            return r * x + x

        # If using profiling, a different function is needed to test different
        # shapes, or we'll use a cached script.
        def fn_test_rand2(x, y):
            # [FUNCTION DOC] HELPER: Variant of the random function with x squared, defined
            # separately to avoid a cached script.
            r = torch.rand_like(y)
            return r * x * x

        x = torch.randn(4, 4, dtype=torch.float, device="cuda")
        y = torch.randn(4, 4, dtype=torch.float, device="cuda")
        script_f = torch.jit.script(fn_test_rand)
        warmup_forward(script_f, x, y)
        out = script_f(x, y)
        self.assertAllFused(script_f.graph_for(x, y))
        x.requires_grad_(True)
        out = script_f(x, y)
        self.assertAllFused(
            script_f.graph_for(x, y),
            except_for=(
                "aten::size",
                "prim::BroadcastSizes",
                "aten::_size_if_not_equal",
            ),
        )

        # test that broadcasting random produces correct results
        x = torch.ones(4, 4, dtype=torch.float, device="cuda")
        y = torch.ones(4, dtype=torch.float, device="cuda")
        script_f = torch.jit.script(fn_test_rand2)
        warmup_forward(script_f, x, y)
        out = script_f(x, y)
        self.assertEqual(out[0, :] + torch.zeros(4, 4, device="cuda"), out)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skip("rand_like is not supported yet")
    def test_rand_diamond(self):
        # [FUNCTION DOC] Skipped CUDA test checking a random value used on two branches of a
        # diamond cancels so the output equals x plus y.
        def fn_test_diamond(x, y):
            # [FUNCTION DOC] HELPER: Adds a random tensor to x, subtracts it from y and
            # returns the sum of both.
            r = torch.rand_like(y)
            a = x + r
            b = y - r
            return a + b

        x = torch.randn(4, 4, dtype=torch.float, device="cuda")
        y = torch.randn(4, 4, dtype=torch.float, device="cuda")
        script_f = torch.jit.script(fn_test_diamond)
        warmup_forward(script_f, x, y)
        out = script_f(x, y)
        self.assertEqual(out, x + y)

    def test_scalar(self):
        # [FUNCTION DOC] Checks a function mixing a zero-dim scalar tensor operand is
        # scripted correctly and fully fused on cpu.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns two times x plus y.
            return 2 * x + y

        x = torch.tensor(0.1, dtype=torch.float, device="cpu")
        y = torch.tensor(1, dtype=torch.float, device="cpu")
        ge = self.checkScript(fn, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skip("https://github.com/pytorch/pytorch/issues/156438")
    def test_inlined_optimized_graph(self):
        # [FUNCTION DOC] Skipped test (linked issue) that runs a scripted relu function with
        # three different input ranks, then checks the last optimized graph has one If and
        # TensorExpr groups, repeated after inlining.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function returning relu of x plus x.
            return torch.relu(x + x)

        for _ in range(3):
            foo(torch.rand([4, 4]))

        for _ in range(3):
            foo(torch.rand([10]))

        for _ in range(3):
            foo(torch.rand([2, 2, 2]))

        g = torch.jit.last_executed_optimized_graph()

        FileCheck().check_count("prim::If", 1, exactly=True).check(
            "prim::TensorExpr"
        ).run(g)
        torch._C._jit_pass_inline(g)
        f = FileCheck()
        for _ in range(3):
            f.check("prim::If").check("prim::TensorExpr")
        f.run(g)

    def test_small_constant(self):
        # [FUNCTION DOC] On each device checks tiny and huge float constants in a traced
        # expression are fused.
        for device in self.devices:

            def fn_test_small_constant(x, y):
                # [FUNCTION DOC] HELPER: Combines x and y with tiny constants and scales up
                # by one hundred million.
                return (1e-8 * x + 5e-9 * y) * 1e8

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(fn_test_small_constant, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    # Currently we don't pull constants into fusion groups, because in some
    # cases it could remove the constant from the original graph and now our
    # fusion group needs to return that constant for its other users.
    # Instead of never pulling constants into the fusion group, we should just
    # be more careful at how we rewrite its users.
    # TODO: fix that and reenable the test.
    def test_tensor_scalar_ops(self):
        # [FUNCTION DOC] Checks that a constant scalar is fused with surrounding ops, and a
        # non-constant scalar input is also fused, with correct results when the scalar
        # value changes.
        for device in self.devices:

            def should_fuse(x):
                # [FUNCTION DOC] HELPER: Adds a constant to x and multiplies by x.
                z = 3.0
                y = x + z
                return x * y

            def should_fuse_scalar(x, z):
                # [FUNCTION DOC] HELPER: Adds an integer made from the scalar z to x and
                # multiplies by x.
                y = x + int(z)
                return x * y

            inputs = [torch.randn(2, 2, dtype=torch.float, device=device)]
            ge = self.checkScript(should_fuse, inputs)
            graph = ge.graph_for(*inputs)
            fusion_groups = self.findFusionGroups(graph)
            self.assertEqual(len(fusion_groups), 1)
            FileCheck().check("aten::add").check("aten::mul").run(str(fusion_groups[0]))

            inputs = [
                torch.randn(2, 2, dtype=torch.float, device=device),
                torch.tensor(3.0, dtype=torch.float, device=device),
            ]
            ge = self.checkScript(should_fuse_scalar, inputs)
            # Check that the fused graph computes correct results when the scalar
            # input changes.
            inputs = [
                torch.randn(2, 2, dtype=torch.float, device=device),
                torch.tensor(7.0, dtype=torch.float, device=device),
            ]
            self.assertEqual(ge(*inputs), should_fuse_scalar(*inputs))
            # The TE fuser supports fusion of non-constant scalars
            self.assertGraphContainsExactly(
                ge.graph_for(*inputs), FUSION_GROUP, 1, consider_subgraphs=True
            )

    def test_where_and_typing(self):
        # [FUNCTION DOC] On each device checks that a function returning a boolean mask and
        # a where result is fully fused except tuple construction.
        for device in self.devices:

            def f(x, y):
                # [FUNCTION DOC] HELPER: Returns mask x greater than y and where(mask, x,
                # y).
                mask = x > y
                res = torch.where(mask, x, y)
                return mask, res

            x = torch.randn(4, 4, dtype=torch.double, device=device)
            y = torch.randn(4, 4, dtype=torch.double, device=device)

            script_f = self.checkScript(f, (x, y))
            self.assertAllFused(
                script_f.graph_for(x, y), except_for={"prim::TupleConstruct"}
            )

    def test_disabled(self):
        # [FUNCTION DOC] Disables CPU fusion, scripts a small function and checks no fusion
        # groups exist, then restores the previous CPU fusion state (restore is skipped if
        # an assertion fails).
        old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        torch._C._jit_override_can_fuse_on_cpu(False)

        def fn(a):
            # [FUNCTION DOC] HELPER: Returns a squared plus a.
            return a**2 + a

        x = torch.randn(4, dtype=torch.float, device="cpu")
        s = self.checkScript(fn, (x,))
        g = s.graph_for(x)
        self.assertEqual(len(self.findFusionGroups(g)), 0)

        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuser_state)

    def data_for(self, dtype, device="cuda", size=None):
        # [FUNCTION DOC] HELPER: Builds test data. With no size makes a two-element float
        # tensor of one and two, otherwise a random tensor of the given size. Converts to
        # boolean by comparing with two, quantizes for quantized dtypes (scale 0.1, zero
        # point 1), otherwise casts to the dtype.
        if size is None:
            v = torch.arange(1, 3, dtype=torch.float, device=device)
        else:
            v = torch.rand(*size, device=device)
        if dtype == torch.bool:
            return v > 2
        elif dtype in [torch.qint8, torch.quint8, torch.qint32]:
            return torch.quantize_per_tensor(v, 0.1, 1, dtype=dtype)
        else:
            return v.to(dtype)

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/121876")
    def test_torch_to(self):
        # [FUNCTION DOC] Checks the fuser handles type conversion: no-op, non-constant
        # dtype, pinned memory and cross-device conversions are not fused, while after
        # freezing a module with constant target dtype all supported dtype pairs are fused
        # (float16 on cpu skipped). Skipped for TorchDynamo.
        # test no op
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted conversion to float, a no-op.
            return x.to(torch.float)

        foo(torch.tensor([3.0], dtype=torch.float))
        foo(torch.tensor([3.0], dtype=torch.float))
        FileCheck().check_not("TensorExpr").run(
            torch.jit.last_executed_optimized_graph()
        )

        # test not fusing non-const inputs
        @torch.jit.script
        def foo(x, dtype: int):
            # [FUNCTION DOC] HELPER: Scripted conversion with a non-constant dtype.
            return x.to(dtype)

        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        FileCheck().check_not("TensorExpr").run(
            torch.jit.last_executed_optimized_graph()
        )

        # test not fusing to_pinned inputs
        @torch.jit.script
        def foo(x, dtype: int):
            # [FUNCTION DOC] HELPER: Scripted conversion requesting pinned memory.
            return x.to(pin_memory=True)

        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        FileCheck().check_not("TensorExpr").run(
            torch.jit.last_executed_optimized_graph()
        )

        # test across-device not supported
        if torch.cuda.is_available():

            @torch.jit.script
            def foo(x):
                # [FUNCTION DOC] HELPER: Scripted conversion to cuda device.
                return x.to(device="cuda")

            foo(torch.tensor([3.0], dtype=torch.float))
            foo(torch.tensor([3.0], dtype=torch.float))
            FileCheck().check_not("TensorExpr").run(
                torch.jit.last_executed_optimized_graph()
            )

        sizes = [(1, 4), (4, 4)]
        # reuses cast impl, smaller dtype set for faster test
        dtypes = [
            torch.bool,
            torch.int,
            torch.float16,
            torch.float32,
            torch.float64,
        ]

        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that converts its input to a stored dtype.
            def __init__(self, dtype):
                # [FUNCTION DOC] HELPER: Stores the target dtype after calling the parent
                # initializer.
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x converted to the stored dtype.
                return x.to(self.dtype)

        bad_dtypes = []
        for dtype, output_dtype, device, size in product(
            dtypes, dtypes, self.devices, sizes
        ):
            # TODO: Add back when https://github.com/pytorch/pytorch/issues/55905 is closed
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            if dtype == output_dtype:
                continue

            x = self.data_for(dtype, device, size=size)
            mod = MyMod(output_dtype)
            ref = mod.forward(x)
            # use freezing to make non-Tensor args to `to` constant
            mod = torch.jit.freeze(torch.jit.script(mod.eval()))
            warmup_forward(mod.forward, x)
            self.assertEqual(ref, mod.forward(x))
            self.assertLastGraphAllFused()

    @unittest.skip("Temporarily disabled")
    def test_masked_fill(self):
        # [FUNCTION DOC] Skipped test of masked_fill over integer, float and bool dtypes,
        # devices, scalar values and sizes, tracing and checking close and fusion.
        # KNOWN ISSUE: the failure message builder uses a name op that is not defined in
        # this scope (a noqa hides the lint error), so the error path would raise NameError.
        dtypes = [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            # TODO: Add back when https://github.com/pytorch/pytorch/issues/55905 is closed
            # torch.float16,
            torch.float32,
            torch.float64,
            torch.bool,
        ]
        sizes = [(2,), (4, 4)]
        for self_dtype, device, scalar_val, size in product(
            dtypes, self.devices, [0.4, 3], sizes
        ):
            input_v = self.data_for(self_dtype, device, size=size)
            mask = self.data_for(torch.bool, device, size=size)

            def fn(input_v, mask):
                # [FUNCTION DOC] HELPER: Fills input where mask is set with the captured
                # scalar value.
                return torch.masked_fill(input_v, mask, scalar_val)

            ref = fn(input_v, mask)
            try:
                t = torch.jit.trace(fn, (input_v, mask))
                torch.testing.assert_close(ref, t(input_v, mask))
                self.assertLastGraphAllFused()
            except Exception as e:
                raise RuntimeError(
                    " ".join(
                        [
                            "Failed:",
                            str(self_dtype),
                            op.__name__,  # noqa: F821
                            device,
                            str(size),
                        ]
                    )
                ) from e

    def test_isnan(self):
        # [FUNCTION DOC] Traces an isnan lambda over inputs containing NaN, all device and
        # many dtypes (half on cpu skipped), warms up and checks results equal eager and the
        # graph is fully fused; failures raise RuntimeError naming dtype and device.
        x = torch.rand([4])
        x[0] = float("nan")
        inputs = [x, torch.tensor([float("nan"), 0.5])]
        dtypes = [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bool,
        ]

        for inp, device, dtype in product(inputs, self.devices, dtypes):
            # TODO: Add back when https://github.com/pytorch/pytorch/issues/55905 is closed
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            inp = inp.to(device=device, dtype=dtype)
            try:
                f = torch.jit.trace(lambda x: x.isnan(), (inp,))
                warmup_forward(f, inp)
                self.assertEqual(f(inp), inp.isnan())
                self.assertLastGraphAllFused()
            except Exception as e:
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), "isnan", device])
                ) from e

    def test_gelu(self):
        # [FUNCTION DOC] Checks traced GELU with a boolean approximate flag over dtypes,
        # devices and sizes against eager, skipping combinations eager rejects, and checks
        # full fusion. Raises RuntimeError on failure.
        def apply(fn):
            # [FUNCTION DOC] HELPER: Returns a function calling the given op with x and
            # approximate.
            return lambda x, approximate: fn(x, approximate)

        unary_ops = [
            F.gelu,
        ]
        sizes = [(1,), (2,), (4, 4)]
        for dtype, op, device, size in product(
            self.dtypes, unary_ops, self.devices, sizes
        ):
            # TODO: Add back when https://github.com/pytorch/pytorch/issues/55905 is closed
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                x = self.data_for(dtype, device, size=size)
                cond = self.data_for(torch.bool, device)
                fn = apply(op)
                ref = fn(x, cond)
            except Exception:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (x, cond))
                torch.testing.assert_close(ref, t(x, cond))
                self.assertAllFused(t.graph_for(x, cond))
            except Exception as e:
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device, str(size)])
                ) from e

    def test_unary_ops(self):
        # [FUNCTION DOC] Large sweep of unary ops (math and activation functions) over
        # dtypes, devices and sizes: records eager result, skips combos eager rejects and
        # gpu-only ops on cpu, traces, compares closeness and checks full fusion; failures
        # re-raised as RuntimeError. Runs with emit hooks disabled.
        with torch._jit_internal._disable_emit_hooks():

            def apply(fn):
                # [FUNCTION DOC] HELPER: Returns a function applying the unary op.
                return lambda x: fn(x)

            unary_ops = [
                torch.lgamma,
                torch.sigmoid,
                torch.reciprocal,
                torch.neg,
                torch.relu,
                F.relu6,
                torch.log,
                torch.log10,
                torch.log1p,
                torch.log2,
                torch.exp,
                torch.expm1,
                torch.erf,
                torch.erfc,
                torch.cos,
                torch.sin,
                torch.tan,
                torch.acos,
                torch.asin,
                torch.cosh,
                torch.sinh,
                torch.atan,
                torch.tanh,
                F.hardtanh,
                F.hardsigmoid,
                F.hardswish,
                F.softplus,
                F.silu,
                F.mish,
                F.elu,
                torch.sqrt,
                torch.rsqrt,
                torch.abs,
                # TODO broken on int8 since
                # https://github.com/pytorch/pytorch/pull/85144
                # RuntimeError: Invalid integral op_type: 23
                # torch.ceil,
                # torch.floor,
                # torch.round,
                # torch.trunc,
                torch.frac,
                # TODO: broken on ROCm?
                # F.hardshrink,
                F.leaky_relu,
                lambda x: torch.threshold(x, 0, -10),
                # TODO: broken since type promotion was added
                # lambda x: torch.clamp(x, -10, 10),
            ]
            gpu_only = {torch.erf, torch.erfc}
            sizes = [(1,), (2,), (4, 4)]
            for dtype, op, device, size in product(
                self.dtypes, unary_ops, self.devices, sizes
            ):
                # TODO: Add back when https://github.com/pytorch/pytorch/issues/55905 is closed
                if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                    continue
                # todo - re-enable. fails with .500
                if dtype == torch.bfloat16 and op == torch.round:
                    continue
                if op in gpu_only and device == "cpu":
                    continue
                try:
                    x = self.data_for(dtype, device, size=size)
                    fn = apply(op)
                    ref = fn(x)
                except Exception:
                    # If eager mode doesn't support a dtype/op/device combo,
                    # neither does the fuser.  Catch everything to avoid needing to
                    # guess what errors might be thrown by eager.
                    continue
                try:
                    t = torch.jit.trace(fn, (x,))
                    torch.testing.assert_close(ref, t(x))
                    self.assertAllFused(t.graph_for(x))
                except Exception as e:
                    raise RuntimeError(
                        " ".join(
                            ["Failed:", str(dtype), op.__name__, device, str(size)]
                        )
                    ) from e

    def test_binary_ops(self):
        # [FUNCTION DOC] Sweep of binary ops over dtypes and devices: compares traced to
        # eager and checks full fusion (float-only ops are checked only for floating
        # dtypes). Combos that fail in eager are skipped; other failures raise RuntimeError.
        def apply(fn):
            # [FUNCTION DOC] HELPER: Returns a function applying the binary op to x and y.
            return lambda x, y: fn(x, y)

        binary_ops = [
            operator.__and__,
            operator.__or__,
            operator.__xor__,
            torch.add,
            torch.sub,
            torch.mul,
            torch.min,
            torch.max,
            lambda x, y: torch.lerp(x, y, 0.5),
            torch.atan2,
            torch.div,
            torch.eq,
            torch.ne,
            torch.ge,
            torch.gt,
            torch.lt,
            torch.fmod,
            torch.remainder,
            lambda x, y: y.type_as(x),
        ]
        fp_only = [
            torch.fmod,
            torch.remainder,
        ]
        devices = self.devices
        for dtype, op, device in product(self.dtypes, binary_ops, devices):
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                fn = apply(op)
                ref = fn(x, y)
            except Exception:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (x, y))
                self.assertEqual(ref, t(x, y))
                if op not in fp_only or dtype.is_floating_point:
                    self.assertAllFused(t.graph_for(x, y))
            except Exception as e:
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    def test_binary_scalar_ops(self):
        # [FUNCTION DOC] Builds IR text for each binary scalar op with int, float and bool
        # operand types, interprets it, then compiles with TensorExprKernel and compares
        # kernel results with the interpreter. Skips IR that cannot be interpreted; raises
        # RuntimeError if compile or run fails.
        def apply(fn):
            # [FUNCTION DOC] HELPER: Returns a function applying the binary op; unused in
            # the body.
            return lambda x, y: fn(x, y)

        ir_template = """
        graph(%x : {dtype_x}, %y : {dtype_y}):
          %z = {op}(%x, %y)
          return (%z)"""

        binary_ops = [
            "aten::mul",
            "aten::add",
            "aten::sub",
            "aten::div",
            "aten::lt",
            "aten::le",
            "aten::eq",
            "aten::ne",
            "aten::gt",
            "aten::ge",
            "aten::__or__",
            "aten::__xor__",
            "aten::__and__",
            "aten::__lshift__",
            "aten::__rshift__",
        ]
        dtypes = ["int", "float", "bool"]
        values = {"int": [10, 3], "float": [12.34, 2.78], "bool": [True, False]}
        for dtype_x, dtype_y, op in product(dtypes, dtypes, binary_ops):
            code = ir_template.format(dtype_x=dtype_x, dtype_y=dtype_y, op=op)

            # Interpret the graph
            try:
                graph = torch._C.parse_ir(code)
                for x, y in product(values[dtype_x], values[dtype_y]):
                    ref = torch._C._jit_interpret_graph(graph, (x, y))
            except Exception:
                # If we can't interpret this IR, don't bother checking NNC.
                continue

            # Compile the graph
            try:
                k = torch._C._te.TensorExprKernel(graph)
            except Exception as e:
                raise RuntimeError(" ".join(["Compilation failed:", str(code)])) from e

            # Run the graph
            for x, y in product(values[dtype_x], values[dtype_y]):
                ref = torch._C._jit_interpret_graph(graph, (x, y))
                try:
                    res = k.run((x, y))
                    self.assertEqual(ref, res)
                except Exception as e:
                    raise RuntimeError(
                        " ".join(["Failed at runtime:", str(x), str(y), str(code)])
                    ) from e

    def test_matmul(self):
        # [FUNCTION DOC] Checks matmul traced for many shape pairs on cpu against eager;
        # only the two-dimensional case is checked for fusion, others only for correctness.
        # Skipped for dynamic shapes.
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the matrix product of x and y.
            return torch.matmul(x, y)

        devices = ["cpu"]  # No cuda support for ext calls yet
        sizes = [
            [[128, 128], [128, 128]],
            [[10, 10], [10, 10]],
            [[1, 16], [16, 128]],
            [[128], [128]],
            [[128], [128, 128]],
            [[3], [3]],
            [[3, 4], [4]],
            [[10, 3, 4], [4]],
            [[10, 3, 4], [10, 4, 5]],
            [[10, 3, 4], [4, 5]],
        ]

        # Only 2D x 2D matrix multiply is supported. For non-supported sizes we
        # still want to run results verification to test that we didn't
        # accidentally fuse it, but we skip the 'is-fused' check.
        # TODO: add support for other shape combinations and make this set empty:
        skip_is_fused_check_sizes = [
            "[[128], [128]]",
            "[[128], [128, 128]]",
            "[[3], [3]]",
            "[[3, 4], [4]]",
            "[[10, 3, 4], [4]]",
            "[[10, 3, 4], [10, 4, 5]]",
            "[[10, 3, 4], [4, 5]]",
        ]
        for dtype, size, device in product(self.dtypes, sizes, devices):
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                size_x, size_y = size
                x = self.data_for(dtype, device, size=size_x)
                y = self.data_for(dtype, device, size=size_y)
                ref = fn(x, y)
            except Exception as e:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (x, y))
                t(x, y)
                self.assertEqual(ref, t(x, y))
                if str(size) not in skip_is_fused_check_sizes:
                    self.assertAllFused(t.graph_for(x, y))
            except Exception as e:
                raise RuntimeError(" ".join(["Failed:", str(dtype), device])) from e

    def test_binary_tensor_scalar_ops(self):
        # [FUNCTION DOC] Sweep of binary tensor-scalar ops over dtypes, devices and scalars:
        # compares traced to eager and checks full fusion, with emit hooks disabled.
        # KNOWN ISSUE: tracing passes the bare tensor in parentheses rather than a tuple, so
        # the input is not wrapped in a one-element tuple.
        with torch._jit_internal._disable_emit_hooks():

            def apply_with_scalar(fn, scalar):
                # [FUNCTION DOC] HELPER: Returns a function applying the op to x and a
                # captured scalar.
                return lambda x: fn(x, scalar)

            # FIXME: Fails in IR Eval: torch.int64 and_ cpu
            binary_ops = [
                operator.__and__,
                operator.__or__,
                operator.__xor__,
                torch.add,
                torch.sub,
                torch.mul,
                torch.eq,
                torch.ne,
                torch.ge,
                torch.lt,
                torch.gt,
            ]
            devices = self.devices
            # Maybe we should split this into separate tests to speed it up by
            # only using  scalar values relevant to particular ops
            scalars = [1.5, 3, 0, -2.0, -1]
            for dtype, op, device, scalar in product(
                self.dtypes, binary_ops, devices, scalars
            ):
                if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                    continue
                try:
                    x = self.data_for(dtype, device)
                    fn = apply_with_scalar(op, scalar)
                    ref = fn(x)
                except Exception:
                    # If eager mode doesn't support a dtype/op/device combo,
                    # neither does the fuser.  Catch everything to avoid needing to
                    # guess what errors might be thrown by eager.
                    continue
                try:
                    t = torch.jit.trace(fn, (x))
                    self.assertEqual(ref, t(x))
                    self.assertAllFused(t.graph_for(x))
                except Exception as e:
                    raise RuntimeError(
                        " ".join(["Failed:", str(dtype), op.__name__, device])
                    ) from e

    def test_binary_div_ops(self):
        # [FUNCTION DOC] Like the tensor-scalar sweep for div, remainder and fmod with
        # nonzero scalars: checks traced output equals eager but does not check fusion.
        def apply_with_scalar(fn, scalar):
            # [FUNCTION DOC] HELPER: Returns a function applying the op to x and a captured
            # scalar.
            return lambda x: fn(x, scalar)

        binary_ops = [
            torch.div,
            torch.remainder,
            torch.fmod,
        ]
        devices = self.devices
        # Maybe we should split this into separate tests to speed it up by
        # only using  scalar values relevant to particular ops
        scalars = [1.5, 3, -2.0, -1]  # skip 0
        for dtype, op, device, scalar in product(
            self.dtypes, binary_ops, devices, scalars
        ):
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                x = self.data_for(dtype, device)
                fn = apply_with_scalar(op, scalar)
                ref = fn(x)
            except Exception:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (x))
                self.assertEqual(ref, t(x))
            except Exception as e:
                raise RuntimeError(
                    f"Failed: {dtype} {op.__name__} {device} {scalar}"
                ) from e

    def test_binary_pow(self):
        # [FUNCTION DOC] Checks pow with scalar exponents for float32 and float64 on each
        # device: traced results equal eager and graph is fully fused.
        def apply_with_scalar(fn, scalar):
            # [FUNCTION DOC] HELPER: Returns a function applying the op to x and a captured
            # scalar.
            return lambda x: fn(x, scalar)

        dtypes = [
            # FIXME: 'pow' fails with dtype=torch.float16/device=cuda/scalar=0
            # torch.float16,
            torch.float32,
            torch.float64,
            # torch.bool intentionally not included
        ]
        binary_ops = [
            torch.pow,
        ]
        # Maybe we should split this into separate tests to speed it up by
        # only using  scalar values relevant to particular ops
        scalars = [1.5, 3, 0, -2.0, -1]
        for dtype, op, device, scalar in product(
            dtypes, binary_ops, self.devices, scalars
        ):
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                x = self.data_for(dtype, device)
                fn = apply_with_scalar(op, scalar)
                ref = fn(x)
            except Exception:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (x))
                self.assertEqual(ref, t(x))
                self.assertAllFused(t.graph_for(x))
            except Exception as e:
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    def test_ternary_ops(self):
        # [FUNCTION DOC] Sweep of lerp and addcmul over dtypes and devices: compares traced
        # with eager and checks full fusion; failures raise RuntimeError.
        def apply(fn):
            # [FUNCTION DOC] HELPER: Returns a function applying a ternary op to x, y and z.
            return lambda x, y, z: fn(x, y, z)

        ternary_ops = [
            torch.lerp,
            torch.addcmul,
        ]
        devices = self.devices
        for dtype, op, device in product(self.dtypes, ternary_ops, devices):
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                z = self.data_for(dtype, device)
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (x, y, z))
                self.assertEqual(ref, t(x, y, z))
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    def test_ternary_norm_ops(self):
        # [FUNCTION DOC] Sweep of batch_norm called with positional tensors over dtypes and
        # devices: compares traced with eager and checks full fusion; failures raise
        # RuntimeError.
        def apply(fn):
            # [FUNCTION DOC] HELPER: Returns a function applying a ternary op to x, y and z.
            return lambda x, y, z: fn(x, y, z)

        ternary_ops = [
            F.batch_norm,
        ]
        devices = self.devices
        for dtype, op, device in product(self.dtypes, ternary_ops, devices):
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                x = self.data_for(dtype, device, size=[5, 3, 128, 128])
                y = self.data_for(dtype, device, size=[3])
                z = self.data_for(dtype, device, size=[3])
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (x, y, z))
                self.assertEqual(ref, t(x, y, z))
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    @unittest.skip(
        "FIXME: fuser doesn't include ListConstruct nodes to the group causing a failure"
    )
    def test_list_ops(self):
        # [FUNCTION DOC] Skipped test (fuser does not pull list construction into groups) of
        # concatenating squares of three inputs over dtypes and devices.
        def apply(fn):
            # [FUNCTION DOC] HELPER: Returns a function concatenating squares of its three
            # inputs with the given op.
            return lambda x, y, z: fn([x * x, y * y, z * z])

        devices = self.devices
        list_ops = [
            torch.cat,
        ]
        for dtype, op, device in product(self.dtypes, list_ops, devices):
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                x = self.data_for(dtype, device, size=[5, 4, 1, 7])
                y = self.data_for(dtype, device, size=[5, 4, 1, 7])
                z = self.data_for(dtype, device, size=[5, 4, 1, 7])
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (x, y, z))
                self.assertEqual(ref, t(x, y, z))
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    def test_where_ops(self):
        # [FUNCTION DOC] Sweep of where variants (tensor, scalar float and scalar int
        # branches) over dtypes and devices with a boolean condition: compares traced to
        # eager and checks full fusion.
        def apply(fn):
            # [FUNCTION DOC] HELPER: Returns a function applying the op to a condition and
            # two tensors.
            return lambda cond, x, y: fn(cond, x, y)

        ops = [
            torch.where,
            lambda cond, x, y: torch.where(cond, x, 3.1415),
            lambda cond, x, y: torch.where(cond, 42, y),
        ]
        devices = self.devices
        for dtype, op, device in product(self.dtypes, ops, devices):
            if dtype in [torch.float16, torch.bfloat16] and device == "cpu":
                continue
            try:
                cond = self.data_for(torch.bool, device)
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                fn = apply(op)
                ref = fn(cond, x, y)
            except Exception:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (cond, x, y))
                self.assertEqual(ref, t(cond, x, y))
                self.assertAllFused(t.graph_for(cond, x, y))
            except Exception as e:
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    def test_unsupported_dtypes(self):
        # [FUNCTION DOC] For unsupported dtypes (unsigned byte, complex, quantized) checks
        # traced output equals eager and no fusion groups exist; skips dtypes eager cannot
        # make.
        for device in self.devices:

            def fn(x):
                # [FUNCTION DOC] HELPER: Returns x squared plus x.
                return x * x + x

            unsupported_dtypes = [
                torch.uint8,
                torch.complex32,
                torch.complex64,
                torch.complex128,
                torch.qint8,
                torch.quint8,
                torch.qint32,
            ]
            for dtype in unsupported_dtypes:
                try:
                    x = self.data_for(dtype, device)
                    ref = fn(x)
                except Exception:
                    # If eager mode doesn't support a dtype/op/device combo,
                    # neither does the fuser.  Catch everything to avoid needing to
                    # guess what errors might be thrown by eager.
                    continue
                t = torch.jit.trace(fn, (x,))
                self.assertEqual(ref, t(x))
                self.assertEqual(len(self.findFusionGroups(t.graph_for(x))), 0)

    def test_superslomo(self):
        # [FUNCTION DOC] Super-SloMo derived test with strided mixed-size inputs and mixed-
        # shape outputs; scripts a function, runs it four times, compares to eager and
        # checks full fusion. The cpu device is removed when LLVM is not available.
        devices = self.devices.copy()
        if not LLVM_ENABLED:
            devices.remove("cpu")
        for device in devices:
            # Test extracted from Super-SloMo: https://github.com/avinashpaliwal/Super-SloMo
            # A few interesting things happen here: strided inputs of mixed size,
            # plus outputs of mixed shapes.  The latter characteristic happened to
            # expose a memory corruption bug due to not properly guarding the
            # outputs.
            def eager(t0, t1, t2, t3, t4):
                # [FUNCTION DOC] HELPER: Multiplies and adds five input tensors, returning a
                # tuple of four results.
                t5 = torch.mul(t0, t4)
                t6 = torch.mul(t2, t3)
                t7 = torch.mul(t6, t1)
                t9 = torch.add(t5, t7)
                t11 = torch.add(t0, t6)
                ft_p = torch.div(t9, t11)
                return (ft_p, t11, t9, t6)

            t0 = torch.rand(1, 6, 352, 352, device=device).transpose(0, 1)
            t1 = torch.rand(6, 3, 352, 352, device=device)
            t2 = torch.rand(6, device=device)[None, None, None, :].permute(3, 0, 1, 2)
            t3 = torch.rand(6, 1, 352, 352, device=device)
            t4 = torch.rand(6, 3, 352, 352, device=device)
            inputs = [t0, t1, t2, t3, t4]

            script = torch.jit.script(eager)
            for _ in range(4):
                for pair in zip(script(*inputs), eager(*inputs)):
                    test, ref = pair
                    torch.testing.assert_close(test, ref)
                    self.assertAllFused(
                        script.graph_for(*inputs), except_for={"prim::TupleConstruct"}
                    )

    def test_sub_gt_and(self):
        # [FUNCTION DOC] Checks a function whose boolean mask is used in a never-executed
        # branch still gives correct scripted results on each device.
        for device in self.devices:

            def eager(t1, t2, t3, t4, t: float):
                # [FUNCTION DOC] HELPER: Builds a boolean mask from two thresholded
                # differences, verifies its dtype, returns k plus one if t is large else w.
                # Raises AssertionError if the mask dtype is not bool.
                w = t1 - t2
                h = t3 - t4
                k = (w > t) & (h > t)
                if k.dtype != torch.bool:
                    raise AssertionError("k.dtype should be bool")
                if t > 0.5:
                    # Putting a use of k in a never-executed conditional prevents
                    # profiling its type, which leaves it as "Tensor".  If we
                    # propagate Tensor back to the definition of k, we have to be
                    # careful not to create a fusion group containing it.
                    return k + 1
                return w

            t = torch.rand(8, dtype=torch.float, device=device)
            scripted = self.checkScript(eager, (t, t, t, t, 0.1))

    @skipIfTorchDynamo("too slow")
    def test_chunk_mul_one(self):
        # [FUNCTION DOC] Checks chunking a large tensor and scaling one piece scripts
        # correctly on each device. Skipped for dynamic shapes and for TorchDynamo.
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:

            def eager(x):
                # [FUNCTION DOC] HELPER: Chunks x in three along the last dim and returns
                # the first piece times three plus the other two.
                z, y, w = torch.chunk(x, 3, -1)
                return z * 3, y, w

            x = torch.rand(64, 1, 3072, dtype=torch.float, device=device)
            z, y, w = eager(x)
            script = self.checkScript(eager, (x,))

    def test_eq_unsqueeze_type_as(self):
        # [FUNCTION DOC] Checks equality mask, unsqueeze and type_as scripts correctly on
        # each device.
        for device in self.devices:

            def eager(a, b):
                # [FUNCTION DOC] HELPER: Returns the mask b equal one unsqueezed and
                # converted to a's type, plus the mask.
                mask = b == 1
                mask = torch.unsqueeze(mask, -1)
                x = mask.type_as(a)
                return x, mask

            a = torch.rand(1, 64, 1024, device=device, dtype=torch.float)
            b = torch.randint(-2, 2, (1, 64), device=device, dtype=torch.long)
            script = self.checkScript(eager, (a, b))

    def test_neg_pow(self):
        # [FUNCTION DOC] Checks negated pow for tensor-tensor, tensor-scalar and scalar-
        # tensor operands scripts correctly; fusion checks are commented out.
        def eager_tt(a: torch.Tensor, b: torch.Tensor):
            # [FUNCTION DOC] HELPER: Negated power of two tensors.
            return torch.neg(torch.pow(a, b))

        def eager_ts(a: torch.Tensor, b: float):
            # [FUNCTION DOC] HELPER: Negated power of a tensor and a float.
            return torch.neg(torch.pow(a, b))

        def eager_st(a: float, b: torch.Tensor):
            # [FUNCTION DOC] HELPER: Negated power of a float and a tensor.
            return torch.neg(torch.pow(a, b))

        a = torch.rand(1, dtype=torch.float)
        b = torch.rand(1, dtype=torch.float)
        s = b.item()
        script = self.checkScript(eager_tt, (a, b))
        # TODO: re-enable fusion, which doesn't work right now. just test correctness for now
        # self.assertAllFused(script.graph_for(a, b))
        script = self.checkScript(eager_ts, (a, s))
        # self.assertAllFused(script.graph_for(a, s))
        script = self.checkScript(eager_st, (s, b))
        # self.assertAllFused(script.graph_for(s, b))

    @unittest.skipIf(not LLVM_ENABLED, "Too slow to run with the TE interpreter")
    def test_conv2d_depthwise(self):
        # [FUNCTION DOC] Depthwise 2D convolution scripted and checked for full fusion.
        # Skipped unless LLVM is enabled and for dynamic shapes.
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        def eager(input, weight, bias):
            # [FUNCTION DOC] HELPER: Depthwise conv2d with stride one, padding one and 72
            # groups.
            return torch.conv2d(input, weight, bias, stride=1, padding=1, groups=72)

        input = torch.rand((1, 72, 56, 56), dtype=torch.float)
        weight = torch.rand((72, 1, 3, 3), dtype=torch.float)
        bias = torch.rand((72), dtype=torch.float)

        script = self.checkScript(eager, (input, weight, bias))
        self.assertAllFused(script.graph_for(input, weight, bias))

    def test_conv2d(self):
        # [FUNCTION DOC] Checks a regular 2D convolution scripts correctly and is not fused
        # into TensorExpr. Skipped for dynamic shapes.
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        def eager(input, weight, bias):
            # [FUNCTION DOC] HELPER: conv2d with stride one, padding one and one group.
            return torch.conv2d(input, weight, bias, stride=1, padding=1, groups=1)

        input = torch.rand((1, 64, 56, 56), dtype=torch.float)
        weight = torch.rand((64, 64, 3, 3), dtype=torch.float)
        bias = torch.rand((64), dtype=torch.float)

        script = self.checkScript(eager, (input, weight, bias))
        FileCheck().check_not("TensorExpr").run(
            torch.jit.last_executed_optimized_graph()
        )

    def test_type_as_cat(self):
        # [FUNCTION DOC] With inlined fusion groups, traces concat of x with a type_as
        # converted tensor over dtype pairs (excluding half types) and checks results and
        # full fusion.
        with inline_fusion_groups():

            def eager(x, y):
                # [FUNCTION DOC] HELPER: Concatenates x with y converted to x type along dim
                # one.
                return torch.cat((x, y.type_as(x)), dim=1)

            dtypes = self.dtypes.copy()
            # CPU fuser doesn't support float16.
            dtypes.remove(torch.float16)
            dtypes.remove(torch.bfloat16)
            for dtype1, dtype2 in product(dtypes, dtypes):
                x = torch.randint(2, (1, 13)).to(dtype1)
                zero = torch.tensor([[0]]).to(dtype2)
                one = torch.tensor([[1]]).to(dtype2)
                script = torch.jit.trace(eager, (x, zero))
                for _ in range(3):
                    torch.testing.assert_close(script(x, zero), eager(x, zero))
                    torch.testing.assert_close(script(x, one), eager(x, one))
                self.assertAllFused(script.graph_for(x, one))

    def test_to_device(self):
        # [FUNCTION DOC] Checks conversion to the cpu device followed by relu is scripted
        # and fully fused.
        def eager(x):
            # [FUNCTION DOC] HELPER: Converts to cpu and applies relu.
            return x.to(device="cpu").relu()

        x = torch.rand(8)
        script = self.checkScript(eager, (x,))
        self.assertAllFused(script.graph_for(x))

    def test_dims(self):
        # [FUNCTION DOC] Checks division of a strided three-dimensional tensor by a one
        # element tensor sum is fully fused.
        def eager(x, y):
            # [FUNCTION DOC] HELPER: Divides x by y plus a small epsilon.
            return x / (y + 0.0001)

        x = torch.linspace(-1, 1, 768, dtype=torch.float32).as_strided(
            (1, 1, 768), (768, 1, 1)
        )
        y = torch.tensor([[[2.0]]], dtype=torch.float32)
        script = self.checkScript(eager, (x, y))
        self.assertAllFused(script.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_channels_last_dims_dynamic(self):
        # [FUNCTION DOC] CUDA test: for each subset of dims set to size one in a channels-
        # last input, with dynamic strategy, traces an add and checks output equals eager,
        # remains channels-last and contains TensorExpr.
        def eager(x, y):
            # [FUNCTION DOC] HELPER: Adds x to y plus a small epsilon.
            return x + (y + 0.0001)

        indices = [0, 1, 2, 3]
        sets = []
        for i in range(len(indices) + 1):
            for subset in combinations(indices, i):
                sets.append(subset)  # noqa: PERF402

        for set in sets:
            size = [2, 3, 4, 5]
            for index in set:
                size[index] = 1
            inp = torch.rand(size).to(memory_format=torch.channels_last).cuda()
            with texpr_enable_strategy([("DYNAMIC", 20)]):
                foo_s = torch.jit.trace(eager, (inp, inp))
                for _ in range(3):
                    out = foo_s(inp, inp)
                out_eager = eager(inp, inp)
                self.assertEqual(out_eager, out)
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
                g = torch.jit.last_executed_optimized_graph()
                FileCheck().check("TensorExpr").run(g)

    def test_exhaust_specializations(self):
        # [FUNCTION DOC] With a static strategy limited to one specialization, runs a
        # scripted function with two shapes and checks the inlined graph has exactly two
        # TensorExpr nodes.
        with texpr_enable_strategy([("STATIC", 1)]):

            @torch.jit.script
            def foo(x):
                # [FUNCTION DOC] HELPER: Scripted function returning x three times added.
                return x + x + x

            for _ in range(3):
                foo(torch.rand([2, 2]))

            for _ in range(3):
                foo(torch.rand([4, 4, 4]))

            g = torch.jit.last_executed_optimized_graph()
            torch._C._jit_pass_inline(g)

            FileCheck().check_count("TensorExpr", 2, exactly=True).run(g)

    def test_unsqueeze_var_dim(self):
        # [FUNCTION DOC] Checks unsqueeze with a non-constant dimension argument on a
        # permuted input scripts correctly.
        def eager(x, y, z: int):
            # [FUNCTION DOC] HELPER: Multiplies x by y unsqueezed at integer dim z.
            return x * torch.unsqueeze(y, dim=z)

        x = torch.rand(4, 4, 64).permute(1, 0, 2)
        y = torch.rand(4, 4)
        z = 2
        script = self.checkScript(eager, (x, y, z))

    def _test_fwd_bwd(self, fn):
        # [FUNCTION DOC] HELPER: Trains a function eagerly and scripted for 11 steps with
        # gradient descent on arange inputs, using the same random upstream gradient, then
        # asserts final outputs are close.
        x = torch.arange(-10, 10, dtype=torch.float32, requires_grad=True)
        xs = torch.arange(-10, 10, dtype=torch.float32, requires_grad=True)
        script = torch.jit.script(fn)
        for _ in range(11):
            y = fn(x)
            g0 = torch.rand_like(y)
            y.backward(g0)

            ys = script(xs)
            ys.backward(g0)

            with torch.no_grad():
                x -= 0.1 * x.grad
                xs -= 0.1 * xs.grad
                x.grad = None
                xs.grad = None
        torch.testing.assert_close(y, ys)

    def test_relu_fwd_bwd(self):
        # [FUNCTION DOC] Runs forward/backward comparison for scaled relu.
        def eager(x):
            # [FUNCTION DOC] HELPER: Relu of x times 1.01.
            return torch.relu(x * 1.01)

        self._test_fwd_bwd(eager)

    def test_hardswish_fwd_bwd(self):
        # [FUNCTION DOC] Runs forward/backward comparison for scaled hardswish.
        def eager(x):
            # [FUNCTION DOC] HELPER: Hardswish of x times 1.01.
            return F.hardswish(x) * 1.01

        self._test_fwd_bwd(eager)

    def test_hardsigmoid_fwd_bwd(self):
        # [FUNCTION DOC] Runs forward/backward comparison for scaled hardsigmoid.
        def eager(x):
            # [FUNCTION DOC] HELPER: Hardsigmoid of x times 1.01.
            return F.hardsigmoid(x) * 1.01

        self._test_fwd_bwd(eager)

    def test_cat_graph_opt(self):
        # [FUNCTION DOC] Checks log of a concatenation of three tensors is scripted
        # correctly and the last graph is fully fused.
        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Log of the concatenation of three tensors.
            return torch.log(torch.cat([x, y, z]))

        self.checkScript(
            foo, (torch.rand([5, 5]), torch.rand([2, 5]), torch.rand([1, 5]))
        )
        # TODO: not sure why not updated graph isn't reflected in last_optimized_graph
        self.assertLastGraphAllFused()

    def test_dynamic_cat(self):
        # [FUNCTION DOC] With inlined fusion groups, runs a scripted function that
        # concatenates nested concatenations in a loop over lists of varying-sized tensors
        # three times; passes if it does not fail.
        with inline_fusion_groups():

            @torch.jit.script
            def repro(
                xs: list[torch.Tensor], ys: list[torch.Tensor], zs: list[torch.Tensor]
            ):
                # [FUNCTION DOC] HELPER: Scripted list comprehension concatenating x with
                # the concatenation of y and z.
                return [
                    torch.cat([x, torch.cat([y, z], dim=-1)], dim=-1)
                    for x, y, z in zip(xs, ys, zs)
                ]

            for _ in range(3):
                N = 3
                xs = [torch.ones(21) for _ in range(N)]
                # Note: concat of ys and zs will have the same size for each
                # pair, even though the individual ys and zs do not.
                ys = [torch.ones(N - i) for i in range(N)]
                zs = [torch.ones(i) for i in range(N)]
                repro(xs, ys, zs)

    def test_scalar_only_inputs(self):
        # [FUNCTION DOC] Checks a function whose only input is a float scalar scripts
        # correctly.
        def eager(b: float):
            # [FUNCTION DOC] HELPER: Returns a one element ones tensor times b.
            a = torch.ones(1)
            return a * b

        script = self.checkScript(eager, (1.0,))

    def test_cat_2k_args(self):
        # [FUNCTION DOC] With inlined groups, checks a concat of 2000 copies of a tensor
        # followed by relu is traced correctly and creates no fusion groups (too many
        # arguments).
        with inline_fusion_groups():

            def eager(x):
                # [FUNCTION DOC] HELPER: Relu of the concatenation of 2000 copies of x.
                return torch.relu(torch.cat([x for _ in range(2000)]))

            x = torch.randn(1)
            trace = self.checkTrace(eager, (x,))
            fusion_groups = self.findFusionGroups(trace.graph_for(x))
            self.assertEqual(len(fusion_groups), 0)

    def test_adaptive_avg_pool2d(self):
        # [FUNCTION DOC] Compiles traced adaptive average pooling with tuple and int output
        # sizes with TensorExprKernel and compares with the traced output.
        # TODO: once the adaptive_avg_pool2d is available in OpInfo DB, this
        # test should be moved there
        with inline_fusion_groups():

            def foo1(x):
                # [FUNCTION DOC] HELPER: Adaptive average pool to 2 by 2.
                return torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))

            def foo2(x):
                # [FUNCTION DOC] HELPER: Adaptive average pool to size 2.
                return torch.nn.functional.adaptive_avg_pool2d(x, (2))

            x = torch.randn(4, 4, 4)
            for foo in [foo1, foo2]:
                f = torch.jit.trace(foo, (x,))
                kernel = torch._C._te.TensorExprKernel(f.graph)
                correct_val = f(x)
                self.assertEqual(kernel.run((x,)), correct_val)

    def test_unrolled_cat(self):
        # [FUNCTION DOC] Checks a loop of concatenations over x rows: after warming up with
        # a size-one input the profile is baked in, and a longer input must still give the
        # eager result.
        with inline_fusion_groups():

            def eager(x):
                # [FUNCTION DOC] HELPER: Builds a tensor by looping over rows of x and
                # appending relu of each row.
                ret = torch.empty(0)
                for i in range(x.shape[0]):
                    ret = torch.cat([ret, x[i].relu()])
                return ret

            script = torch.jit.script(eager)

            # Warm up with size=1 tensor; since the loop iterates once the
            # profile data will be "burned in" assuming size=1, and then
            # unrolled.
            x = torch.ones(1, 1)
            for _ in range(3):
                script(x)

            torch.testing.assert_close(eager(x), script(x))

            # Now when an input hits the unrolled path, it will produce an
            # incorrectly-sized tensor, since size=1 has been burned in.
            x = torch.ones((8, 1))
            torch.testing.assert_close(eager(x), script(x))

    @skipIfTorchDynamo("too slow")
    @unittest.skipIf(TEST_WITH_ASAN, "takes 10+ minutes on asan")
    def test_batch_norm(self):
        # [FUNCTION DOC] Checks batch_norm variants (with/without weight and bias) are fully
        # fused and close to eager on each device. Skipped for TorchDynamo and under ASAN.
        def test(fn, args):
            # [FUNCTION DOC] HELPER: Traces fn with args, asserts full fusion and closeness
            # to eager treating NaN as equal.
            trace = torch.jit.trace(fn, args)
            self.assertAllFused(trace.graph_for(*args))
            torch.testing.assert_close(fn(*args), trace(*args), equal_nan=True)

        def bn(i, x, rv):
            # [FUNCTION DOC] HELPER: Batch norm with weight and bias then relu.
            return torch.batch_norm(i, x, x, x, rv, False, 0.1, 1e-4, False).relu()

        def bn_no_weight(i, x, rv):
            # [FUNCTION DOC] HELPER: Batch norm without weight, then relu.
            return torch.batch_norm(i, None, x, x, rv, False, 0.1, 1e-4, False).relu()

        def bn_no_bias(i, x, rv):
            # [FUNCTION DOC] HELPER: Batch norm without bias, then relu.
            return torch.batch_norm(i, x, None, x, rv, False, 0.1, 1e-4, False).relu()

        def bn_neither(i, x, rv):
            # [FUNCTION DOC] HELPER: Batch norm without weight and bias, then relu.
            return torch.batch_norm(
                i, None, None, x, rv, False, 0.1, 1e-4, False
            ).relu()

        for device in self.devices:
            i = torch.randn(4, 16, 32, 40, device=device)
            x = torch.randn(16, device=device)
            rv = torch.randn(
                16, device=device
            ).abs()  # running_var must be non-negative
            for fn in [bn, bn_no_weight, bn_no_bias, bn_neither]:
                test(fn, (i, x, rv))

    def test_profiler(self):
        # [FUNCTION DOC] Runs a scripted function three times under the autograd profiler
        # and asserts the profile table includes a fused_mul_add entry.
        @torch.jit.script
        def test(x, y, z):
            # [FUNCTION DOC] HELPER: Scripted function returning x times y plus z.
            return x * y + z

        args = [torch.randn(4) for _ in range(3)]
        with torch.autograd.profiler.profile() as prof:
            for _ in range(3):
                test(*args)
        self.assertIn("fused_mul_add", prof.table())

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/156436")
    def test_skip_grad_in_check(self):
        # [FUNCTION DOC] Skipped on Linux (linked issue). Runs a scripted function with
        # requires-grad input under inference mode and checks the inlined graph keeps
        # exactly one If node.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function returning x plus two divided by two.
            return (x + 2) / 2

        inp = torch.rand([4, 4])
        for _ in range(3):
            foo(inp)

        inp.requires_grad_(True)
        with torch.inference_mode():
            for _ in range(3):
                foo(inp)
        g = torch.jit.last_executed_optimized_graph()
        torch._C._jit_pass_inline(g)
        torch._C._jit_pass_inline(g)
        FileCheck().check_count("prim::If", 1, exactly=True).run(g)

    def test_dynamic_shapes(self):
        # [FUNCTION DOC] With a dynamic strategy, traces unary/binary/ternary functions over
        # differently shaped tensors (including transposed and channels last) and checks
        # exactly one dynamic guard in the optimized graph, then checks one guard per
        # generator for repeated shapes.
        from functools import partial

        n = 10

        gen_tensor = (
            lambda n: R(1, n),
            lambda n: R(n, n),
            lambda n: R(n, n).transpose(0, 1),
            lambda n: R(n + 1, n + 1, 2)[:n, n, 0],
            lambda n: R(n, n, 2)[:, :, 0],
            lambda n: R(n, n + 1, n + 2, n + 3).to(memory_format=torch.channels_last),
        )

        with texpr_enable_strategy([("DYNAMIC", 20)]):

            def foo(x, y, z):
                # [FUNCTION DOC] HELPER: Sigmoid of tanh of x; function caching is disabled
                # on it.
                return torch.sigmoid(torch.tanh(x))

            foo.__disable_jit_function_caching__ = True

            def fi(x, y, z):
                # [FUNCTION DOC] HELPER: Tanh of x plus y; function caching disabled.
                return torch.tanh(x + y)

            fi.__disable_jit_function_caching__ = True

            def fum(x, y, z):
                # [FUNCTION DOC] HELPER: Tanh of x plus y plus z; function caching disabled.
                return torch.tanh(x + y) + z

            fum.__disable_jit_function_caching__ = True

            funcs = [foo, fi, fum]
            with inline_fusion_groups():
                for device in self.devices:
                    I = partial(torch.randint, 0, 100, device=device)
                    R = partial(torch.randn, device=device)

                    for i, func in enumerate(funcs):
                        num_args = i + 1
                        for gen in gen_tensor:
                            inps = (gen(n), gen(n), gen(n))
                            func_s = torch.jit.trace(func, inps, check_trace=False)
                            torch._C._jit_pass_erase_shape_information(func_s.graph)
                            for _ in range(2):
                                x, y, z = gen(n), gen(n), gen(n)
                                func_s(x, y, z)

                            for _incr in range(3):
                                func_s(*[gen(n + 1) for _ in range(3)])

                            g = torch.jit.last_executed_optimized_graph()
                            torch._C._jit_pass_inline(g)
                            torch._C._jit_pass_dce(g)

                            # We should see only one optimized kernel
                            FileCheck().check_count(
                                "TensorExprDynamicGuard", 1, exactly=True
                            ).run(g)
                            self.assertEqual(func(*inps), func_s(*inps))

                    gen = gen_tensor[0]
                    inps = (gen(n), gen(n), gen(n))
                    foo_s = torch.jit.trace(foo, inps)
                    torch._C._jit_pass_erase_shape_information(foo_s.graph)
                    g_prev = None
                    for gen in gen_tensor:
                        for i in range(3):
                            foo_s(*[gen(n + i) for _ in range(3)])
                            inps = (gen(n), gen(n), gen(n))
                            self.assertEqual(foo_s(*inps), foo(*inps))
                    g = torch.jit.last_executed_optimized_graph()
                    torch._C._jit_pass_inline(g)
                    torch._C._jit_pass_dce(g)
                    FileCheck().check_count(
                        "TensorExprDynamicGuard", len(gen_tensor), exactly=True
                    ).run(g)

    @unittest.skipIf(not RUN_CUDA, "half-precision NNC fusion requires CUDA")
    def test_autocast_up(self):
        # [FUNCTION DOC] CUDA test checking a script function that converts half to full
        # precision then exps is fused.
        def f(x):
            # [FUNCTION DOC] HELPER: Converts x to full precision and returns its
            # exponential.
            y = x._autocast_to_full_precision(True, True)
            z = torch.exp(y)
            return z

        x = torch.rand((2, 2), dtype=torch.half, device="cuda")
        scr = torch.jit.script(f)
        scr(x)
        scr(x)
        self.assertLastGraphAllFused()

    @unittest.skipIf(not RUN_CUDA, "half-precision NNC fusion requires CUDA")
    def test_autocast_down(self):
        # [FUNCTION DOC] CUDA test checking sigmoid followed by conversion to half precision
        # is fused.
        def f(x):
            # [FUNCTION DOC] HELPER: Sigmoid of x then conversion to reduced precision half.
            y = torch.sigmoid(x)
            z = y._autocast_to_reduced_precision(True, True, torch.half, torch.half)
            return z

        x = torch.rand((2, 2), dtype=torch.float, device="cuda")
        scr = torch.jit.script(f)
        scr(x)
        scr(x)
        self.assertLastGraphAllFused()

    @unittest.skipIf(not LLVM_ENABLED, "Compiles with TensorExprKernel")
    def test_to_dtype(self):
        # [FUNCTION DOC] Checks a chain of precision conversions on float and bfloat16
        # traced inputs: float32 case is fully fused, bfloat16 case yields two groups;
        # values match within tolerance. Requires LLVM.
        def f(x):
            # [FUNCTION DOC] HELPER: Sigmoid, conversions to half-reduced, full precision,
            # then to bfloat16 and float32.
            y = torch.sigmoid(x)
            z = y._autocast_to_reduced_precision(True, True, torch.half, torch.bfloat16)
            h = z._autocast_to_full_precision(True, True)
            i = h.to(dtype=torch.bfloat16)
            j = i.to(dtype=torch.float32)
            return j

        x = torch.rand((2, 2), dtype=torch.float32)
        scr = torch.jit.trace(f, x)
        scr(x)
        scr(x)
        self.assertLastGraphAllFused()
        self.assertEqual(f(x), scr(x), atol=4e-3, rtol=4e-3)

        bf_x = torch.rand((2, 2), dtype=torch.bfloat16)
        bf_scr = torch.jit.trace(f, bf_x)
        bf_scr(bf_x)
        bf_scr(bf_x)
        graph = bf_scr.graph_for(bf_x)
        fusion_groups = self.findFusionGroups(graph)
        self.assertEqual(len(fusion_groups), 2)
        self.assertEqual(f(bf_x), bf_scr(bf_x), atol=4e-3, rtol=4e-3)

    def test_with_strict_fusion(self):
        # [FUNCTION DOC] Tests the strict_fusion context: a pure arithmetic function
        # succeeds and fuses; an unfusible random op raises with unfused operators text;
        # calling in eager warns; autodiff and multiple separate fusions also raise errors.
        # KNOWN ISSUE: the error assertions quote source text from scripted nested
        # functions, so their doc blocks are placed above the def lines to leave quoted text
        # unchanged.
        # [FUNCTION DOC] HELPER: Scripted function adding x three times within a strict
        # fusion region.
        def success(x):
            with torch.jit.strict_fusion():
                return x + x + x

        scripted = self.checkScript(success, (torch.rand([4]),))
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check_not("aten::add").check("prim::TensorExprGroup").run(g)

        # [FUNCTION DOC] HELPER: Strict fusion region with an unfusible random tensor
        # addition; scripting it should fail with source quoted unfused operators.
        def foo(x):
            with torch.jit.strict_fusion():
                return x + x + torch.rand([4]) + 3

        with self.assertRaises(Exception) as error_out:
            foo_s = torch.jit.script(foo)
            foo_s(torch.rand([4]))
            foo_s(torch.rand([4]))
            print(torch.jit.last_executed_optimized_graph())
        fc = FileCheck().check("Found unfused operators")
        fc.check("aten::rand(SymInt[] size")
        fc.check("torch.rand([4]").run(str(error_out.exception))

        with warnings.catch_warnings(record=True) as warns:
            foo(torch.rand([4]))

        FileCheck().check("Only works in script mode").run(str(warns[0]))

        # [FUNCTION DOC] HELPER: Strict fusion region with random tensor plus x three times
        # for the autodiff failure case.
        def test_autodiff(x):
            with torch.jit.strict_fusion():
                return torch.rand([4]) + x + x + x

        foo_s = torch.jit.script(test_autodiff)
        inp = torch.rand([4], requires_grad=True)
        with self.assertRaises(Exception) as error_out:
            for _ in range(3):
                foo_s(inp)
        f = FileCheck().check("unfused operators").check("aten::rand")
        f.run(str(error_out.exception))

        # [FUNCTION DOC] HELPER: Two separate strict fusion regions in one return, expected
        # to raise a multiple fusions error.
        def test_separate_fusions(x, y):
            with torch.jit.strict_fusion():
                return x + x + x, y + y + y

        inp = torch.rand([4], requires_grad=True)
        with self.assertRaises(Exception) as error_out:
            for _ in range(3):
                foo_s = torch.jit.script(test_separate_fusions)
                foo_s(inp, inp)

        f = FileCheck().check("Found multiple fusions")
        f.run(str(error_out.exception))

    def test_constant_chunk_shapes(self):
        # [FUNCTION DOC] Regression test that building shape expressions for a ConstantChunk
        # whose input is built from unsupported constants does not fail: traces the
        # function, checks one ConstantChunk in the group and runs it four times matching
        # eager. Skipped for dynamic shapes.
        # We had an issue where buildShapeExpressions would fail as show below:
        #
        # %1 : Tensor = Constant[..]  # not supported, we don't build this shape
        # %2 : Tensor = Constant[..]  # not supported
        # %3 : Tensor = aten::add(%1, %2)  # inputs not supported, we don't build shape
        # ... = prim::ConstantChunk[..](%3)  # it forgets to check whether input shapes exist, and fails
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:

            def f(x, y):
                # [FUNCTION DOC] HELPER: Adds x, y and a constant tensor then chunks and
                # multiplies the halves.
                r = torch.tensor(4)
                z1, z2 = (x + y + r).chunk(2, dim=1)
                return z1 * z2

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(f, (x, y))
            graph = ge.graph_for(x, y)

            # make sure that we are actually testing the right scenario
            FileCheck().check("with " + FUSION_GROUP + "_").check_count(
                "ConstantChunk", 1, exactly=True
            ).run(str(graph))

            f_traced = torch.jit.trace(f, (x, y))

            for _ in range(4):
                # make sure this doesn't error out
                res = f_traced(x, y)

            self.assertEqual(res, f(x, y))

    @unittest.skipIf(not RUN_CUDA_HALF, "half-precision NNC fusion requires CUDA")
    def test_pow_multiple_dtype(self):
        # [FUNCTION DOC] CUDA half test: a scripted sigmoid then power with float exponent
        # on half input runs four times and matches eager.
        # https://github.com/pytorch/pytorch/issues/75476
        def fn(p: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Sigmoid then power with gamma.
            p = torch.sigmoid(p)
            result = p**gamma
            return result

        x = torch.rand((2, 2), dtype=torch.half, device="cuda")

        ref = fn(x)

        script_fn = torch.jit.script(fn)
        for _ in range(4):
            res = script_fn(x)

        self.assertEqual(ref, res)


class TestTEFuserStatic(TestTEFuser):
    # [CLASS DOC] Runs all TestTEFuser tests with static shapes (class attribute
    # dynamic_shapes is False).
    dynamic_shapes = False


class TestTEFuserDynamic(TestTEFuser):
    # [CLASS DOC] Runs all TestTEFuser tests with dynamic shapes (class attribute
    # dynamic_shapes is True).
    dynamic_shapes = True


del TestTEFuser

works_list = [
    "__radd__",
    "__rdiv__",
    "__rmul__",
    "__rmod__",
    "abs",
    "acos",
    "add",
    "addcmul",
    "addmm.decomposed",
    "asin",
    "atan",
    "atan2",
    "ceil",
    "clamp",
    "clamp.scalar",
    "contiguous",
    "cos",
    "cosh",
    "div.no_rounding_mode",
    "div.true_rounding",
    "div.floor_rounding",
    "div.trunc_rounding",
    "eq",
    "erf",
    "erfc",
    "exp",
    "expand",
    "expand_as",
    "expm1",
    "floor",
    "fmod",
    "fmod.autodiffed",
    "ge",
    "gt",
    "isnan",
    "le",
    "lerp",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "lt",
    "masked_fill",
    "max.binary",
    "mean",
    "min.binary",
    "mm",
    "mul",
    "ne",
    "neg",
    "nn.functional.hardshrink",
    "nn.functional.hardsigmoid",
    "nn.functional.hardswish",
    "nn.functional.softplus",
    "nn.functional.hardtanh",
    "nn.functional.leaky_relu",
    "nn.functional.relu",
    "nn.functional.relu6",
    "nn.functional.softsign",
    "nn.functional.tanhshrink",
    "nn.functional.threshold",
    "permute",
    "pow",
    "reciprocal",
    "remainder",
    "remainder.autodiffed",
    "reshape",
    "reshape_as",
    "round",
    "rsub",
    "rsub.rsub_tensor",
    "rsqrt",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "sub",
    "sum",
    "t",
    "tan",
    "tanh",
    "transpose",
    "true_divide",
    "trunc",
    "unsqueeze",
    "view",
    "view_as",
    "where",
    "bool",
    "byte",
    "char",
    "double",
    "float",
    "half",
    "int",
    "long",
    "short",
    "bool.channels_last",
    "byte.channels_last",
    "char.channels_last",
    "double.channels_last",
    "float.channels_last",
    "half.channels_last",
    "int.channels_last",
    "long.channels_last",
    "short.channels_last",
]

known_failures = [
    "__rmatmul__",
    "frac",
    "matmul",
]

# If your OpInfo test causes this test to fail, add it here
skip_ops = ["conj"]


def get_name(op):
    # [FUNCTION DOC] HELPER: Returns the OpInfo name, joined with the variant test name by a
    # dot when that variant name is non-empty.
    l = [op.name]
    if op.variant_test_name != "":
        l.append(op.variant_test_name)
    return ".".join(l)


# Purpose of this class is to allow super() calls.
# super() [with no arguments] fails, presumably because of how instantiate_device_type_tests works.
# super(TestNNCOpInfo, self) fails because TestNNCOpInfo gets deleted from global scope.
# super(JitCommonTestCase, self).fn() would skip JitCommonTestCase.fn() implementation
class TestNNCOpInfoParent(JitCommonTestCase):
    # [CLASS DOC] HELPER: Empty parent class allowing explicit super calls from
    # TestNNCOpInfo, which cannot use zero-argument super because of device-type
    # instantiation.
    pass


class TestNNCOpInfo(TestNNCOpInfoParent):
    # [CLASS DOC] OpInfo-driven tests (instantiated per device type, cpu and cuda unless
    # FBCODE) that compile op traces with TensorExprKernel and compare run and fallback
    # results with eager, plus a slow correctness test of traced variants. Uses the module
    # lists of working ops, known failures and skipped ops.
    def setUp(self):
        # [FUNCTION DOC] Calls the grandparent setUp and applies the TensorExpr test
        # options.
        super(TestNNCOpInfoParent, self).setUp()
        self.tensorexpr_options = TensorExprTestOptions()

    def tearDown(self):
        # [FUNCTION DOC] Restores the TensorExpr test options and calls the grandparent
        # tearDown.
        self.tensorexpr_options.restore()
        super(TestNNCOpInfoParent, self).tearDown()

    def te_compile(self, device, dtype, op):
        # [FUNCTION DOC] HELPER: For all sample inputs of the op (skipping ops listed in
        # skip_ops), generates function source taking tensor arguments, executes it to
        # define a function calling the op, traces it, builds a TensorExprKernel and asserts
        # its run and fallback outputs equal eager. Mutates the function module name.
        if op.name in skip_ops:
            return
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in sample_inputs_itr:
            arg_values = [sample_input.input] + list(sample_input.args)
            kwarg_values = sample_input.kwargs
            param_names = []
            param_values = []
            fx_args = []
            for idx, v in enumerate(arg_values):
                if isinstance(v, torch.Tensor):
                    param_names.append(f"arg_{idx}")
                    param_values.append(v)
                    fx_args.append(param_names[-1])
                else:
                    fx_args.append(f"{repr(v)}")

            for k, v in kwarg_values.items():
                if isinstance(v, torch.Tensor):
                    param_names.append(k)
                    param_values.append(v)
                    fx_args.append(f"{k} = {k}")
                else:
                    fx_args.append(f"{k} = {repr(v)}")

            code = f"""
def f({", ".join(param_names)}):
    return op.op({", ".join(fx_args)})"""
            g = {"torch": torch, "inf": math.inf, "op": op}
            exec(code, g)
            f = g["f"]
            f.__module__ = "test"
            out = f(*param_values)

            ts_g = torch.jit.trace(f, param_values)
            kernel = torch._C._te.TensorExprKernel(ts_g.graph)
            correct_val = f(*param_values)
            self.assertEqual(kernel.run(tuple(param_values)), correct_val)
            self.assertEqual(kernel.fallback(tuple(param_values)), correct_val)

    @onlyCPU
    @unittest.skipIf(not LLVM_ENABLED, "Compiles with TensorExprKernel")
    @ops(
        [op for op in op_db if get_name(op) in works_list],
        allowed_dtypes=(torch.float,),
    )
    def test_working(self, device, dtype, op):
        # [FUNCTION DOC] Compiles and checks every OpInfo whose name is in the works list
        # (cpu only, requires LLVM).
        self.te_compile(device, dtype, op)

    @onlyCPU
    @unittest.skipIf(not LLVM_ENABLED, "Compiles with TensorExprKernel")
    @ops(
        [op for op in op_db if get_name(op) in known_failures],
        allowed_dtypes=(torch.float,),
    )
    def test_failures(self, device, dtype, op):
        # [FUNCTION DOC] For ops listed as known failures, expects compilation to raise;
        # raises RuntimeError if it unexpectedly succeeds.
        try:
            self.te_compile(device, dtype, op)
        except Exception as e:
            pass
        else:
            raise RuntimeError(
                "Expected test to fail. If it now works, move op into works_list"
            )

    @onlyCPU
    @unittest.skipIf(not LLVM_ENABLED, "Compiles with TensorExprKernel")
    @ops(
        [op for op in op_db if get_name(op) not in works_list + known_failures],
        allowed_dtypes=(torch.float,),
    )
    def test_unsupported(self, device, dtype, op):
        # [FUNCTION DOC] For ops in neither list, expects compilation to raise (tracer
        # warnings ignored) and raises RuntimeError if it works, prompting a move to the
        # works list. Ops in the skip list return early.
        # KNOWN ISSUE: the tracer warning class referenced is not imported here, a noqa
        # hides the lint error.
        if get_name(op) in skip_ops:
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", TracerWarning)  # noqa: F821
                self.te_compile(device, dtype, op)
        except Exception as e:
            pass
        else:
            raise RuntimeError(
                "Expected test to fail. If it now works, move op into works_list"
            )

    @slowTest
    @onlyCPU
    @ops(
        [op for op in op_db if get_name(op) not in known_failures],
        dtypes=OpDTypes.supported,
    )
    def test_nnc_correctness(self, device, dtype, op):
        # [FUNCTION DOC] Slow test comparing traced variants of each supported op with eager
        # results over sample inputs (looser tolerance for bfloat16), then drops all
        # functions of the global compilation unit to limit memory. Skips ops without
        # tracing support.
        if not op.supports_tracing:
            self.skipTest("Requires tracing support")

        with NoTracerWarnContextManager() as no_warn:
            variant_sample_pairs = get_traced_sample_variant_pairs(device, dtype, op)

            for variant, sample in variant_sample_pairs:
                trace = create_traced_fn(self, variant, cache_traced_fn=True)
                ref = variant(
                    *clone_inputs((sample.input, *sample.args)), **sample.kwargs
                )

                trace(*clone_inputs((sample.input, *sample.args)), **sample.kwargs)
                val = trace(
                    *clone_inputs((sample.input, *sample.args)), **sample.kwargs
                )

                atol = 2e-1 if dtype == torch.bfloat16 else 1e-5
                rtol = 2e-1 if dtype == torch.bfloat16 else 1e-5
                self.assertEqual(ref, val, atol=atol, rtol=rtol)

            # https://github.com/pytorch/pytorch/issues/35600
            # each torch.jit.trace adds state to the _python_cu compilation unit
            # since this test traces a lot of functions, out-of-memory can occur
            # if the CU is not cleared.
            torch.jit._state._python_cu.drop_all_functions()


# CPU fuser not currently used in fbcode
only_for = ("cuda") if IS_FBCODE else ("cpu", "cuda")
instantiate_device_type_tests(TestNNCOpInfo, globals(), only_for=only_for)


# Purpose of this class is to allow super() calls. (See TestNNCOpInfoParent)
class TestLoopnestRandomizationParent(JitTestCase):
    # [CLASS DOC] HELPER: Empty parent class allowing explicit super calls from
    # TestLoopnestRandomization.
    pass


class TestLoopnestRandomization(TestLoopnestRandomizationParent):
    # [CLASS DOC] Tests running the TE fuser with loopnest random transformation seed
    # enabled via environment variable; setUp saves and overrides fuser flags, tearDown
    # restores them.
    def setUp(self):
        # [FUNCTION DOC] Saves current CPU/GPU fuser, LLVM, profiling, inlining and texpr
        # fuser states and then enables CPU and GPU fusion, profiling executor, no inlining,
        # texpr fuser, disables forced LLVM, and sets the random transform seed environment
        # variable to one.
        super(TestLoopnestRandomizationParent, self).setUp()
        self.old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        self.old_must_use_cpu_state = torch._C._jit_get_te_must_use_llvm_cpu()
        self.old_gpu_fuser_state = torch._C._jit_can_fuse_on_gpu()

        torch._C._jit_override_can_fuse_on_cpu(True)
        # TODO: force LLVM. need to add it to asan, mac, windows builds + sandcastle
        # torch._C._jit_set_te_must_use_llvm_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

        self.old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        self.old_profiling_mode = torch._C._get_graph_executor_optimize(True)

        self.old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)

        self.texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
        torch._C._jit_set_texpr_fuser_enabled(True)

        self.old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)

        # Set the seed to 1. This tests the codepath through random
        # transformation.
        os.environ["PYTORCH_TENSOREXPR_RANDOM_TRANSFORM_SEED"] = "1"

    def tearDown(self):
        # [FUNCTION DOC] Restores all saved fuser and executor states, resets the random
        # transform seed environment variable to zero and calls the grandparent tearDown.
        torch._C._jit_set_profiling_executor(self.old_profiling_executor)
        torch._C._get_graph_executor_optimize(self.old_profiling_mode)

        torch._C._jit_override_can_fuse_on_gpu(self.old_gpu_fuser_state)
        torch._C._jit_override_can_fuse_on_cpu(self.old_cpu_fuser_state)
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_must_use_cpu_state)
        torch._C._debug_set_fusion_group_inlining(self.old_fusion_inlining)

        torch._C._jit_set_texpr_fuser_enabled(self.texpr_fuser_state)
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_te_must_use_llvm_cpu)

        # Set it back to 0.
        os.environ["PYTORCH_TENSOREXPR_RANDOM_TRANSFORM_SEED"] = "0"
        super(TestLoopnestRandomizationParent, self).tearDown()

    @onlyCPU
    @unittest.skipIf(not LLVM_ENABLED, "Compiles with TensorExprKernel")
    def test_relu(self, device):
        # [FUNCTION DOC] CPU-only test (requires LLVM) tracing a relu expression and raising
        # AssertionError if the traced result is not close to eager.
        def fn_test_relu(x, y):
            # [FUNCTION DOC] HELPER: Relu of x plus half of y.
            return F.relu(x + 0.5 * y)

        x = torch.randn(4, 4, dtype=torch.float, device=device)
        y = torch.randn(4, 4, dtype=torch.float, device=device)

        fn = fn_test_relu
        traced_fn = torch.jit.trace(fn, (x, y))

        ref = fn(x, y)
        res = traced_fn(x, y)
        if not torch.allclose(ref, res):
            raise AssertionError("traced function output does not match reference")


instantiate_device_type_tests(TestLoopnestRandomization, globals(), only_for=("cpu"))


if __name__ == "__main__":
    if sys.version_info < (3, 14):
        run_tests()
