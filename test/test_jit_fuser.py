# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for the TorchScript fuser (fusion groups) on CUDA and, where enabled, CPU.
# Each test scripts or traces a small function, then checks the optimized graph holds exactly one
# fusion group (or a specific node pattern) and that outputs and gradients match eager mode.
# Import-time side effects: when run as a script it parses command line args first so the graph
# executor mode is known, and in profiling mode it switches the C++ JIT to the profiling executor
# and profiling mode. Also imports LSTM cell fixtures from test_jit. Most tests are skipped without
# CUDA; the whole class is skipped under the legacy executor. The test_jit_fuser_legacy module
# re-runs these tests under the legacy executor.

import unittest
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck
from unittest import skipIf

if __name__ == "__main__":
    from torch.testing._internal.common_utils import parse_cmd_line_args

    # The value of GRAPH_EXECUTOR depends on command line arguments so make sure they're parsed
    # before instantiating tests.
    parse_cmd_line_args()

from torch.testing._internal.common_utils import run_tests, IS_SANDCASTLE, ProfilingMode, GRAPH_EXECUTOR, \
    enable_profiling_mode_for_profiling_tests, IS_WINDOWS, TemporaryDirectoryName, shell
from torch.testing._internal.jit_utils import JitTestCase, enable_cpu_fuser, _inline_everything, \
    RUN_CUDA, RUN_CUDA_HALF, RUN_CUDA_MULTI_GPU, warmup_backward
from textwrap import dedent
from itertools import product, permutations
from torch.testing._internal.common_cuda import with_tf32_off

from test_jit import backward_graph, all_backward_graphs, get_lstm_inputs, get_milstm_inputs, \
    LSTMCellC, LSTMCellF, LSTMCellS, MiLSTMCell

if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)


def strip_profiling_nodes(nodes):
    # [FUNCTION DOC] HELPER: Returns a new list of the given graph nodes with profiling bookkeeping
    # nodes (kinds for the bailout template and bailout) removed; the input is not modified.
    profiling_opcodes = {'prim::BailoutTemplate', 'prim::BailOut'}
    return [n for n in nodes if n.kind() not in profiling_opcodes]


def warmup_forward(f, *args):
    # [FUNCTION DOC] HELPER: Calls f with args twice (so the profiling executor can record) and
    # returns the result of the last call.
    profiling_count = 2
    for _ in range(profiling_count):
        results = f(*args)

    return results


@skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, "skip due to SIGIOT failures, #67646")
class TestFuser(JitTestCase):
    # [CLASS DOC] Test case for the fuser built on JitTestCase. Skipped entirely when
    # the graph executor is the legacy one (known crashes). Holds the shared graph
    # assertion helper assertAllFused, a few static function fixtures reused across
    # tests, and CUDA/CPU fusion tests.
    def assertAllFused(self, graph, except_for=()):
        # [FUNCTION DOC] HELPER: Asserts the graph is fully fused. If a
        # differentiable-graph node exists there must be exactly one and its subgraph
        # is inspected instead. Every node must be a constant, fusion group, bailout,
        # tuple construct, or a kind named in except_for, and exactly one fusion group
        # must be present. Raises an assertion error showing the graph on failure.

        diff_graphs = [n for n in graph.nodes() if n.kind() == 'prim::DifferentiableGraph']
        if len(diff_graphs) > 0:
            self.assertEqual(len(diff_graphs), 1)
            graph = diff_graphs[0].g('Subgraph')

        allowed_nodes = {'prim::Constant', 'prim::FusionGroup', 'prim::BailoutTemplate',
                         'prim::BailOut', 'prim::TupleConstruct'} | set(except_for)
        self.assertTrue(all(node.kind() in allowed_nodes for node in graph.nodes()),
                        lambda msg: f'{msg}\ngot {graph}')
        self.assertTrue([node.kind() for node in graph.nodes()].count('prim::FusionGroup') == 1)

    def _test_fused_abs(self, device='cpu'):
        # [FUNCTION DOC] HELPER: Scripts a function that takes the absolute value and doubles it on
        # a random five-element tensor on the given device (default cpu), checks it against eager
        # via checkScript, and asserts the graph is fully fused.
        def func(x):
            # [FUNCTION DOC] HELPER: Fixture computing twice the absolute value of its input.
            return x.abs() * 2

        a = torch.randn(5, device=device)
        scripted = self.checkScript(func, (a,))
        self.assertAllFused(scripted.graph_for(a))

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_abs_cpu(self):
        # [FUNCTION DOC] Runs the fused abs check on CPU with the CPU fuser enabled;
        # skipped on Sandcastle.
        self._test_fused_abs()

    @unittest.skipIf(not IS_WINDOWS, "This is meant to be Windows-specific")
    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_abs_cpu_unicode_temp_dir(self):
        # [FUNCTION DOC] Windows-only: runs test_abs_cpu in a subprocess whose TMP
        # environment variable points at a temp directory with non-ASCII characters in
        # its name, forwarding the legacy executor flag if present in the command line,
        # and asserts exit code zero. Skipped elsewhere.
        with TemporaryDirectoryName(suffix='\u4e2d\u6587') as dname:
            shell_env = os.environ.copy()
            shell_env['TMP'] = dname
            cmd = [sys.executable, os.path.basename(__file__), type(self).__name__ + '.test_abs_cpu']
            legacy_jit_flag = '--jit-executor=legacy'
            for v in sys.argv:
                if v == legacy_jit_flag:
                    cmd.append(legacy_jit_flag)
            return_code = shell(cmd, cwd=os.path.dirname(__file__), env=shell_env)
            self.assertEqual(return_code, 0)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_abs_cuda(self):
        # [FUNCTION DOC] Runs the fused abs check on a CUDA device; skipped without CUDA.
        self._test_fused_abs(device="cuda")

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_zero_element_tensors(self):
        # [FUNCTION DOC] Checks that scripting a function (arctangent of two float-cast inputs)
        # works for zero-element CUDA tensors by comparing script and eager via checkScript. No
        # fusion assertions.
        def decode(sin_t, cos_t):
            # [FUNCTION DOC] HELPER: Fixture returning the two-argument arctangent of its inputs
            # cast to float.
            theta = torch.atan2(sin_t.float(), cos_t.float())
            return theta

        sin = torch.zeros(0, device="cuda")
        cos = torch.zeros(0, device="cuda")
        inputs = [sin, cos]
        self.checkScript(decode, inputs)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_arg_configurations_smoke_cuda(self):
        # [FUNCTION DOC] Smoke test that a traced function (add, chunk in two along dim 1,
        # multiply) gives the same result for a contiguous and a transposed non-contiguous input,
        # so one kernel is not wrongly shared across layouts. Needs CUDA.
        # A smoke test to make sure we won't use the same kernel for contiguous
        # and non-contiguous arguments.
        # TODO: add optionally enabled debug counters to the fuser to verify
        #       that we really can tell the difference between configurations
        def f(x, y):
            # [FUNCTION DOC] HELPER: Fixture adding x and y, chunking the sum into two along dim 1
            # and multiplying the halves.
            z1, z2 = (x + y).chunk(2, dim=1)
            return z1 * z2

        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        traced_f = torch.jit.trace(f, (x, y,))
        self.assertEqual(traced_f(x.t().contiguous(), y), traced_f(x.t(), y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_broadcast_cuda(self):
        # [FUNCTION DOC] Traces a broadcast scale-and-shift on CUDA tensors (a vector scale and
        # shift against a matrix) and asserts the graph is fully fused.
        def scaleshift(x, scale, shift):
            # [FUNCTION DOC] HELPER: Fixture computing x times scale plus shift.
            return x * scale + shift

        inputs = [
            torch.randn(4, 4, dtype=torch.float, device='cuda'),
            torch.randn(4, dtype=torch.float, device='cuda'),
            torch.randn(4, dtype=torch.float, device='cuda'),
        ]
        ge = self.checkTrace(scaleshift, inputs)
        self.assertAllFused(ge.graph_for(*inputs))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no bfloat support with profiling on")
    def test_cuda_bfloat16(self):
        # [FUNCTION DOC] Scripts add followed by relu on a large bfloat16 CUDA tensor and asserts
        # full fusion. Only runs with the legacy executor.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Fixture computing the relu of x plus y.
            return (x + y).relu()
        m = torch.jit.script(foo)
        x = torch.randn(65536).cuda().bfloat16()
        y = torch.randn_like(x)
        self.assertAllFused(m.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_HALF, "no half support")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no half support with profiling on")
    def test_cuda_half(self):
        # [FUNCTION DOC] For each of three half-precision-capable functions (comparison, relu, exp
        # fixtures), traces on half CUDA inputs without trace checking and compares outputs against
        # a float reference cast to half, then compares gradients of the summed outputs likewise
        # (unused inputs allowed, graph retained). Needs CUDA half support and the legacy executor.
        x = torch.randn(4, 4, dtype=torch.half, device='cuda')
        y = torch.randn(4, 4, dtype=torch.half, device='cuda')

        funcs = [
            self.fn_test_comparison_gt_lt,
            self.fn_test_relu,
            self.fn_test_exp
        ]

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
                    output.float().sum(), local_inputs, allow_unused=True, retain_graph=True)
                fusion_grads = torch.autograd.grad(
                    fusion_output.sum(), local_fusion_inputs, allow_unused=True, retain_graph=True)
                grads_half = [t.half() for t in grads]
                self.assertEqual(grads_half, fusion_grads)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_checks_cat_inputs(self):
        # [FUNCTION DOC] Checks that a concatenation whose inputs have different broadcastable
        # shapes (2 by 4 and 1 by 4, output 3 by 4) gives the right result when scripted and is
        # fully fused, i.e. the fuser validates map size of all cat inputs. Needs CUDA.
        # We shouldn't treat cat nodes as broadcasting. All their inputs
        # need to be checked for having the same map size, before we can
        # run the kernel.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Fixture concatenating two polynomial expressions of x and y
            # along dim 0.
            return torch.cat([x + 2 * x + x ** 2, y + 4 * y + y ** 3], dim=0)

        # NOTE: y is broadcastable to x, but output of f(x, y) should have
        # shape 3x4, and not 4x4.
        x = torch.randn(2, 4, dtype=torch.float, device='cuda')
        y = torch.randn(1, 4, dtype=torch.float, device='cuda')

        scripted = self.checkScript(f, (x, y))
        self.assertAllFused(scripted.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    def test_remainder_cuda(self):
        # [FUNCTION DOC] Scripts a function that adds one to a remainder then subtracts one on CUDA
        # tensors and asserts full fusion.
        def cuda_rem(x, y):
            # [FUNCTION DOC] HELPER: Fixture computing one plus the remainder of x and
            # y, minus one.
            return 1 + torch.remainder(x, y) - 1

        a = torch.rand([512], dtype=torch.float).cuda()
        b = torch.rand([512], dtype=torch.float).cuda()
        inputs = [a, b]
        ge = self.checkScript(cuda_rem, inputs)
        graph = ge.graph_for(*inputs)
        self.assertAllFused(graph)

    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    def test_chunk_cuda(self):
        # [FUNCTION DOC] Scripts a function that chunks a CUDA tensor in three along dim 1 and
        # combines the parts; asserts full fusion and that the graph text contains a constant-chunk
        # node with the right chunk count and dim.
        def fn(x):
            # [FUNCTION DOC] HELPER: Fixture chunking x in three along dim 1 and returning the
            # first times the second plus the third.
            a, b, c = x.chunk(3, 1)
            return a * b + c

        inputs = [torch.randn(10, 6, dtype=torch.float, device='cuda')]

        ge = self.checkScript(fn, inputs)
        graph = ge.graph_for(*inputs)
        self.assertAllFused(graph)
        FileCheck().check("prim::ConstantChunk[chunks=3, dim=1]").run(str(graph))

    @staticmethod
    def _test_chunk_correctness(self, device='cpu'):
        # [FUNCTION DOC] HELPER: Static helper, invoked with an explicit test case as self. Defines
        # three functions that chunk a tensor in four along dim 0, 1 and 2 and sum the parts, then
        # runs checkScript for each over a splitSize-one tensor, a contiguous tensor and a
        # transposed non-contiguous tensor on the given device (default cpu).
        def chunk_4_0(x):
            # [FUNCTION DOC] HELPER: Fixture summing the four chunks of x along dim 0.
            x0, x1, x2, x3 = x.chunk(4, 0)
            return x0 + x1 + x2 + x3

        def chunk_4_1(x):
            # [FUNCTION DOC] HELPER: Fixture summing the four chunks of x along dim 1.
            x0, x1, x2, x3 = x.chunk(4, 1)
            return x0 + x1 + x2 + x3

        def chunk_4_last(x):
            # [FUNCTION DOC] HELPER: Fixture summing the four chunks of x along dim 2
            # (the last dim).
            x0, x1, x2, x3 = x.chunk(4, 2)
            return x0 + x1 + x2 + x3

        fns = [chunk_4_0, chunk_4_1, chunk_4_last]
        tensors = [
            # splitSize = 1
            torch.randn(4, 4, 4, dtype=torch.float, device=device),

            # contiguous case
            torch.randn(12, 8, 16, dtype=torch.float, device=device),

            # non-contiguous case
            torch.randn(12, 8, 16, dtype=torch.float, device=device).transpose(1, 2),
        ]

        for tensor in tensors:
            for fn in fns:
                self.checkScript(fn, [tensor])

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_chunk_correctness(self):
        # [FUNCTION DOC] Runs the chunk correctness helper on CPU with the CPU fuser enabled;
        # skipped on Sandcastle. Returns the helper result (none).
        return self._test_chunk_correctness(self, 'cpu')

    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    def test_chunk_correctness_cuda(self):
        # [FUNCTION DOC] Runs the chunk correctness helper on CUDA; skipped without CUDA.
        return self._test_chunk_correctness(self, 'cuda')

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_chunk_distributes_cuda(self):
        # [FUNCTION DOC] Traces the add-chunk-multiply function on CUDA and checks with
        # FileCheck that the graph has a broadcast of tensors, a fusion group, and
        # exactly two constant-chunk nodes.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Fixture chunking the sum of x and y in two along dim 1 and
            # multiplying the halves.
            z1, z2 = (x + y).chunk(2, dim=1)
            return z1 * z2

        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(f, (x, y))
        graph = ge.graph_for(x, y)
        FileCheck().check("broadcast_tensors").check('with prim::FusionGroup_') \
            .check_count('ConstantChunk', 2, exactly=True).run(str(graph))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_chunk_motion_deduplicates_inputs(self):
        # [FUNCTION DOC] Checks that when chunk is moved into the fusion group, duplicate inputs
        # are merged: for two functions that chunk a power of the input, the scripted forward graph
        # has exactly one fusion group and the last node's fusion group has exactly one input.
        # Needs CUDA.
        def func1(x):
            # [FUNCTION DOC] HELPER: Fixture chunking the square of x in two and
            # multiplying the halves.
            z = x * x
            z0, z1 = z.chunk(2)
            return z0 * z1

        def func2(x):
            # [FUNCTION DOC] HELPER: Fixture chunking the cube of x in two and
            # multiplying the halves.
            z = x * x * x
            z0, z1 = z.chunk(2)
            return z0 * z1

        inputs = [
            torch.tensor([1.1, 1.2], device='cuda', dtype=torch.float),
        ]
        for func in [func1, func2]:
            module = self.checkScript(func, inputs)
            forward_graph = module.graph_for(*inputs)
            self.assertGraphContainsExactly(forward_graph, 'prim::FusionGroup', 1)
            fusion_group = list(forward_graph.nodes())[-1]
            self.assertEqual(len(list(fusion_group.inputs())), 1)

    @unittest.skipIf(not RUN_CUDA, "No CUDA")
    def test_chunk_multiple_cuda(self):
        # [FUNCTION DOC] Scripts a function with four inputs of different shapes that chunks three
        # of them along different dims, and asserts full fusion, checking the fuser adds extra
        # arguments in the correct order when arguments are used out of order. Needs CUDA.
        # The arguments are intentionally used out of order as a test to see
        # if the fusion compiler adds extra args in the correct order
        def fn(s, x, y, z):
            # [FUNCTION DOC] HELPER: Fixture adding a scalar-like first argument to all chunk parts
            # of x, y and z.
            z1, z2 = z.chunk(2, 2)
            x1, x2, x3 = x.chunk(3, 1)
            y1, y2 = y.chunk(2, 0)
            return s + x1 + x2 + x3 + y1 + y2 + z1 + z2

        inputs = [
            torch.randn(5, 2, 3, dtype=torch.float, device='cuda'),
            torch.randn(5, 6, 3, dtype=torch.float, device='cuda'),
            torch.randn(10, 2, 3, dtype=torch.float, device='cuda'),
            torch.randn(5, 2, 6, dtype=torch.float, device='cuda'),
        ]

        ge = self.checkScript(fn, inputs)
        self.assertAllFused(ge.graph_for(*inputs))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_minmax(self):
        # [FUNCTION DOC] For both elementwise max and min of a doubled input against another
        # tensor, and for pairs including a NaN scalar tensor, scripts the function and asserts
        # full fusion. Needs CUDA.
        def tmax(a, b):
            # [FUNCTION DOC] HELPER: Fixture computing the elementwise maximum of twice a and b.
            return torch.max(2 * a, b)

        def tmin(a, b):
            # [FUNCTION DOC] HELPER: Fixture computing the elementwise minimum of twice a and b.
            return torch.min(2 * a, b)

        a = torch.randn(4, 4, dtype=torch.float, device="cuda")
        b = torch.randn(4, 4, dtype=torch.float, device="cuda")
        nan = torch.tensor(float('nan'), dtype=torch.float, device="cuda")

        for f, inputs in product(
                (tmax, tmin),
                ([a, b], [a, nan], [b, nan])):
            s = self.checkScript(f, inputs)
            self.assertAllFused(s.graph_for(*inputs))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_clamp(self):
        # [FUNCTION DOC] For four clamp variants (both bounds, infinite max, only max, only min)
        # applied to a sum, and for a tensor or NaN second operand, scripts the function with
        # function caching disabled under profiling mode, asserts full forward fusion (allowing
        # size nodes), then warms up backward and asserts the backward graph is fused (allowing
        # float and sum-to-size nodes). Needs CUDA.
        def func2(a, b):
            # [FUNCTION DOC] HELPER: Fixture clamping the sum of a and b between zero and two.
            return torch.clamp(a + b, min=0, max=2)

        def funcInf(a, b):
            # [FUNCTION DOC] HELPER: Fixture clamping the sum of a and b between zero and infinity.
            return torch.clamp(a + b, min=0, max=float('inf'))

        def funcOptMin(a, b):
            # [FUNCTION DOC] HELPER: Fixture clamping the sum of a and b to an upper
            # bound of two only.
            return torch.clamp(a + b, max=2)

        def funcOptMax(a, b):
            # [FUNCTION DOC] HELPER: Fixture clamping the sum of a and b to a lower
            # bound of zero only.
            return torch.clamp(a + b, min=0)

        a = torch.randn(4, 4, dtype=torch.float, device='cuda', requires_grad=True)
        b = torch.randn(4, 4, dtype=torch.float, device='cuda')
        nan = torch.tensor(float('nan'), dtype=torch.float, device='cuda')

        funcs = (func2, funcInf, funcOptMin, funcOptMax)
        for f, inputs in product(funcs, [[a, b], [a, nan]]):
            f.__disable_jit_function_caching__ = True
            inp1, inp2 = inputs
            s = self.checkScript(f, (inp1, inp2), profiling=ProfilingMode.PROFILING)
            self.assertAllFused(s.graph_for(inp1, inp2), except_for={'aten::size', 'aten::_size_if_not_equal'})
            c = s(inp1, inp2)
            with enable_profiling_mode_for_profiling_tests():
                warmup_backward(c.sum())
            graph = backward_graph(s)
            self.assertAllFused(graph, except_for={'aten::Float', 'aten::_grad_sum_to_size'})

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no half support with profiling on")
    def test_dropout(self):
        # [FUNCTION DOC] Scripts dropout followed by relu on a CUDA tensor requiring grad, runs it
        # twice, warms up backward, and asserts the backward graph (bailout nodes skipped) is fully
        # fused except for a division and constants. Needs CUDA and the legacy executor.
        def func(x):
            # [FUNCTION DOC] HELPER: Fixture applying functional dropout then relu.
            x = torch.nn.functional.dropout(x)
            return torch.nn.functional.relu(x)

        a = torch.randn(4, 4, dtype=torch.float, device='cuda', requires_grad=True)
        s = torch.jit.script(func)
        c = s(a)
        c = s(a)
        warmup_backward(c.sum())
        # skip_check to skip extra bailout nodes in between
        graph = backward_graph(s, skip_check=True)
        self.assertAllFused(graph, except_for={'aten::div', 'prim::Constant'})

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_comparison_eq_ne(self):
        # [FUNCTION DOC] Traces a function using equality and inequality comparisons turned into
        # masks and asserts the graph is fully fused.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Fixture building masks from equal-to-zero and
            # not-equal-to-zero of x, multiplying x by each and adding y.
            mask = (x == 0).type_as(x)
            z = x * mask + y
            mask = (x != 0).type_as(x)
            z = z * mask + y
            return z

        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(f, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @staticmethod
    def fn_test_comparison_gt_lt(x, y):
        # [FUNCTION DOC] HELPER: Static fixture like the equality one but using greater-than-zero
        # and less-than-zero masks; shared by the gt/lt test and the half precision test.
        mask = (x > 0).type_as(x)
        z = x * mask + y
        mask = (x < 0).type_as(x)
        z = z * mask + y
        return z

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_comparison_gt_lt_cuda(self):
        # [FUNCTION DOC] Traces the greater/less-than fixture on CUDA and asserts full fusion.
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(self.fn_test_comparison_gt_lt, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_comparison_ge_le_cuda(self):
        # [FUNCTION DOC] Traces a function using greater-or-equal and less-or-equal masks and
        # asserts full fusion; then makes inputs require grad and asserts fusion again, allowing
        # size, broadcast-sizes and size-if-not-equal nodes. Needs CUDA.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Fixture building masks from greater-or-equal-zero and
            # less-or-equal-zero of x, multiplying x by each and adding y.
            mask = (x >= 0).type_as(x)
            z = x * mask + y
            mask = (x <= 0).type_as(x)
            z = z * mask + y
            return z

        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(f, (x, y))
        self.assertAllFused(ge.graph_for(x, y))
        x.requires_grad_(True)
        y.requires_grad_(True)
        self.assertAllFused(ge.graph_for(x, y), except_for=("aten::size", "prim::BroadcastSizes",
                                                            "aten::_size_if_not_equal"))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_addcmul_cuda(self):
        # [FUNCTION DOC] Traces an addcmul with broadcasting operands (unused inputs allowed) on
        # CUDA and asserts full fusion.
        t = torch.randn(1, 4, dtype=torch.float, device='cuda')
        t1 = torch.randn(4, 1, dtype=torch.float, device='cuda')
        t2 = torch.randn(1, 4, dtype=torch.float, device='cuda')

        def foo(t, t1, t2):
            # [FUNCTION DOC] HELPER: Fixture calling addcmul on t with t plus one and t2 with a
            # fixed scalar value.
            return t.addcmul(t + 1, t2, value=0.1)

        ge = self.checkTrace(foo, (t, t1, t2), allow_unused=True)
        graph = ge.graph_for(t, t1, t2)
        self.assertAllFused(graph)

    # TODO: We leak CUDA memory here because the traced graph holds onto a
    # constant-ified tensor. Since the Python-global CompilationUnit is alive
    # until the end of the process, the memory is effectively leaked.
    # Removed `_cuda` suffix from this test which disables leak-checking.
    # If this is a real problem, we'll need to revisit Torchscript Function
    # lifetimes in Python.
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_lerp(self):
        # [FUNCTION DOC] Traces lerp with a scalar weight and with a tensor weight (captured from
        # the enclosing scope) on broadcasting CUDA tensors, asserting full fusion for each. Per
        # the comment above it, the test lacks the cuda suffix on purpose to skip leak checking,
        # since traced graphs hold a constant tensor.
        start = torch.randn(4, 1, dtype=torch.float, device='cuda')
        end = torch.randn(1, 4, dtype=torch.float, device='cuda')
        weight = torch.tensor(0.5, dtype=torch.float, device='cuda')

        # scalar weight overload
        def foo_weight_scalar(start, end):
            # [FUNCTION DOC] HELPER: Fixture lerp of start plus one toward end with a
            # scalar weight.
            return torch.lerp(start + 1, end, 0.5)

        # tensor weight overload
        def foo_weight_tensor(start, end):
            # [FUNCTION DOC] HELPER: Fixture lerp of start plus one toward end with the enclosing
            # tensor weight.
            return torch.lerp(start + 1, end, weight)

        ge_weight_scalar = self.checkTrace(foo_weight_scalar, (start, end))
        graph = ge_weight_scalar.graph_for(start, end)
        self.assertAllFused(graph)

        ge_weight_tensor = self.checkTrace(foo_weight_tensor, (start, end))
        graph = ge_weight_tensor.graph_for(start, end)
        self.assertAllFused(graph)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_concat_cuda(self):
        # [FUNCTION DOC] Traces a function concatenating the sum and product of two
        # CUDA tensors, asserts full fusion, and checks the graph has a fused concat
        # node directly followed by return.
        hx = torch.randn(3, 20, dtype=torch.float, device='cuda')
        cx = torch.randn(3, 20, dtype=torch.float, device='cuda')

        def foo(hx, cx):
            # [FUNCTION DOC] HELPER: Fixture concatenating the sum and product of hx and cx.
            return torch.cat((hx + cx, hx * cx))

        ge = self.checkTrace(foo, (hx, cx))
        graph = ge.graph_for(hx, cx)
        self.assertAllFused(graph)
        FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_concat_invariant_cuda(self):
        # [FUNCTION DOC] Checks the invariant that a fused concat output is not an input to any
        # node inside the fusion group: traces a function that concatenates a sum and difference
        # then adds a third input, asserts fusion except for the trailing add, and checks the fused
        # concat precedes return.
        # Invariant: the output of prim::FusedConcat may
        # not be an input to any node inside the FusionGroup.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Fixture concatenating x plus y with x minus y, then adding z.
            x1 = x + y
            y1 = x - y
            w = torch.cat([x1, y1])
            return w + z

        x = torch.randn(2, 2, dtype=torch.float, device='cuda')
        y = torch.randn(2, 2, dtype=torch.float, device='cuda')
        z = torch.randn(4, 2, dtype=torch.float, device='cuda')
        ge = self.checkTrace(fn, (x, y, z))
        graph = ge.graph_for(x, y, z)
        self.assertAllFused(graph, except_for={'aten::add'})
        FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    @staticmethod
    def fn_test_exp(x, y):
        # [FUNCTION DOC] HELPER: Static fixture computing the exponential of x plus half of y; used
        # by the exp test and the half precision test.
        return (x + .5 * y).exp()

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_exp_cuda(self):
        # [FUNCTION DOC] Traces the exp fixture on CUDA and asserts full fusion.
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(self.fn_test_exp, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "broken with profiling on")
    @torch._jit_internal._disable_emit_hooks_decorator
    @_inline_everything
    def test_fuse_decompose_normalization(self):
        # [FUNCTION DOC] Checks that batch norm and layer norm are decomposed and fused: wraps a
        # norm module in a script module computing y plus relu of the norm of x, compares optimized
        # and non-optimized outputs, checks which normalization ops appear in the optimized and
        # unoptimized graphs, and that the single fusion group contains expected ops. Needs CUDA
        # and the legacy executor; emit hooks are disabled and everything inlined.
        class ResLike(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module storing a given norm module and computing y plus
            # relu of the module applied to x.
            def __init__(self, norm_module):
                # [FUNCTION DOC] HELPER: Stores the given norm module as attribute nm.
                super().__init__()
                self.nm = norm_module

            @torch.jit.script_method
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Script method adding y to the relu of the stored module
                # applied to x.
                return y + torch.relu(self.nm(x))

        def test_norm_decompose(nm, in_opt_graph, not_in_opt_graph, in_fusegraph):
            # [FUNCTION DOC] HELPER: Builds optimized and non-optimized CUDA copies of the wrapper
            # with shared state, runs both under no-grad on random inputs, asserts outputs match
            # within tolerance, that every string in in_opt_graph appears in the optimized graph
            # text, every string in not_in_opt_graph is absent there but present in the unoptimized
            # graph, that exactly one fusion group exists and its subgraph text contains every
            # string in in_fusegraph.
            model = ResLike(nm).cuda()
            model_noopt = ResLike(nm).cuda()
            model_noopt.load_state_dict(model.state_dict())
            x = torch.randn(2, 16, 8, 8, device='cuda')
            y = torch.randn(2, 16, 8, 8, device='cuda')

            # FIXME: We need differentiation for CNNs for this optimization to trigger
            with torch.no_grad():
                out = model(x, y)
                graph = model.graph_for(x, y)
                rep = str(graph)

                with torch.jit.optimized_execution(False):
                    out_noopt = model_noopt(x, y)
                    rep_noopt = str(model_noopt.graph_for(x, y))
                self.assertEqual(out, out_noopt, atol=3e-5)

            # Check that normalization op has really been decomposed
            for node_in_graph in in_opt_graph:
                self.assertIn(node_in_graph, rep)

            for node_not_in_graph in not_in_opt_graph:
                self.assertNotIn(node_not_in_graph, rep)
                self.assertIn(node_not_in_graph, rep_noopt)

            fusion_groups = [node for node in graph.nodes() if node.kind() == 'prim::FusionGroup']
            self.assertEqual(len(fusion_groups), 1)
            fused_graph = str(fusion_groups[0].g('Subgraph'))
            for node_in_fusegraph in in_fusegraph:
                self.assertIn(node_in_fusegraph, fused_graph)

        # test for batchnorm decompose
        bm = nn.BatchNorm2d(16)
        test_norm_decompose(bm, ['aten::batch_norm_update_stats'],
                            ['aten::batch_norm('], ['aten::sqrt'])

        # test for layernorm decompose
        lm = nn.LayerNorm(8)
        test_norm_decompose(lm, ['aten::batch_norm_stats'],
                            ['aten::layer_norm('], ['aten::sub', 'aten::mul', 'aten::add'])

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_threshold(self):
        # [FUNCTION DOC] Scripts a threshold op plus three additions on a CUDA tensor and asserts
        # full fusion.
        def f(x):
            # [FUNCTION DOC] HELPER: Fixture applying threshold to x and adding x three times.
            return torch.threshold(x, 0, -10) + x + x + x

        x = torch.tensor([-1, -0.5, 0, 1, 2, 3], device='cuda')
        scripted = self.checkScript(f, (x,))
        self.assertAllFused(scripted.graph_for(x))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_scalar_arg_cuda(self):
        # [FUNCTION DOC] Checks a scripted function with a float scalar argument (an int value is
        # passed) is fully fused; then with grad required, scripts a second identical function (to
        # avoid bailouts) and asserts fusion allowing size-related nodes. Needs CUDA.
        def fn_test_scalar_arg(x: torch.Tensor, p: float) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fixture multiplying scalar p by x squared plus x.
            return p * (x * x + x)

        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        p = 3
        scripted = self.checkScript(fn_test_scalar_arg, (x, p))
        self.assertAllFused(scripted.graph_for(x, p))

        x.requires_grad_(True)

        # use another function otherwise we will bailout
        # and won't be able to do fused checks
        def fn_test_scalar_arg_requires_grad(x: torch.Tensor, p: float) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Same as the other scalar-argument fixture; separate so the
            # grad case does not hit a bailout.
            return p * (x * x + x)

        scripted = torch.jit.script(fn_test_scalar_arg_requires_grad)
        scripted(x, p)
        self.assertAllFused(scripted.graph_for(x, p), except_for=("aten::size", "prim::BroadcastSizes",
                                                                  "aten::_size_if_not_equal"))

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @unittest.skip("deduplicating introduces aliasing in backward graph's outputs")
    @enable_cpu_fuser
    def test_fuser_deduplication(self):
        # [FUNCTION DOC] Skipped test (backward outputs would alias after deduplication). Intended
        # to check that fused kernel outputs are deduplicated when removing sum-to-size: scripts a
        # sigmoid of a sum, asserts fusion in forward and backward, and that both gradients share
        # data pointer.
        # See that fusion kernel outputs are deduplicated when removing  _grad_sum_to_size in the fuser's compilation
        # see the discussion in PR #14957.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Fixture computing the sigmoid of x plus y.
            return torch.sigmoid(x + y)

        b = torch.randn(5, 5, requires_grad=True)
        a = torch.randn(5, 5, requires_grad=True)
        s = self.checkScript(f, (a, b))
        self.assertAllFused(s.graph_for(a, b), except_for={
                            'aten::size', 'aten::_size_if_not_equal', 'prim::BroadcastSizes'})

        c = s(a, b)
        results = warmup_backward(c.sum(), [a, b])
        ga2, gb2 = results.pop()
        graph = backward_graph(s)
        self.assertAllFused(graph)
        # check that a, b share storage, i.e. were generated as a single output in the fuser
        self.assertEqual(ga2.data_ptr(), gb2.data_ptr())

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    @unittest.skip("temporarily disabled because fusion was restricted in fixing #22833")
    def test_fuser_iou(self):
        # [FUNCTION DOC] Skipped test (fusion was restricted when fixing issue 22833). Intended to
        # check that most of an intersection-over-union computation on unsqueezed box coordinates
        # is fused in forward and backward.
        # KNOWN ISSUE: the fixture computes each area with the same y coordinate subtracted from
        # itself (b1y2 minus b1y2), so areas are always zero; the test is skipped anyway.
        # This checks if most of Intersection over Union is fused.
        # In particular, the backward contains many _grad_sum_to_size.
        def iou(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2):
            # [FUNCTION DOC] HELPER: Fixture computing intersection over union from corner
            # coordinates of two box sets.
            ltx = torch.max(b1x1, b2x1)  # [N,M]
            lty = torch.max(b1y1, b2y1)
            rbx = torch.min(b1x2, b2x2)
            rby = torch.min(b1y2, b2y2)

            w = (rbx - ltx).clamp(min=0, max=float('inf'))  # [N,M]
            h = (rby - lty).clamp(min=0, max=float('inf'))  # [N,M]
            inter = w * h  # [N,M]

            area1 = (b1x2 - b1x1) * (b1y2 - b1y2)  # [N,1]
            area2 = (b2x2 - b2x1) * (b2y2 - b2y2)  # [1,M]
            iou = inter / (area1 + area2 - inter)
            return iou

        box1 = torch.randn(5, 4, requires_grad=True)
        box2 = torch.randn(5, 4, requires_grad=True)
        # unsqueezing can currently not be fused
        b1x1 = box1[:, 0].unsqueeze(1)  # [N,1]
        b1y1 = box1[:, 1].unsqueeze(1)
        b1x2 = box1[:, 2].unsqueeze(1)
        b1y2 = box1[:, 3].unsqueeze(1)
        b2x1 = box2[:, 0].unsqueeze(0)  # [1,N]
        b2y1 = box2[:, 1].unsqueeze(0)
        b2x2 = box2[:, 2].unsqueeze(0)
        b2y2 = box2[:, 3].unsqueeze(0)

        s = self.checkScript(iou, (b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2))
        self.assertAllFused(s.graph_for(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2),
                            except_for={'aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'})

        with enable_profiling_mode_for_profiling_tests(True):
            c = s(b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2)
            warmup_backward(c.sum(), [b1x1, b1y1, b1x2, b1y2, b2x1, b2y1, b2x2, b2y2])
            graph = backward_graph(s)
            self.assertAllFused(graph, except_for={'aten::size', 'prim::BroadcastSizes', 'aten::_size_if_not_equal'})

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    @enable_cpu_fuser
    def test_fusion_reuse_multi_gpu(self):
        # [FUNCTION DOC] Checks that a CPU-scripted fusion can be reused for CUDA inputs on devices
        # zero and one without crashing (different kernels should be compiled). Needs CUDA,
        # multiple GPUs and the CPU fuser.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Fixture computing x times y times x times y.
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

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    @enable_cpu_fuser
    def test_kernel_cache_multi_gpu(self):
        # [FUNCTION DOC] Checks the kernel spec cache: scripts a function with three identical
        # fusion groups on CPU, cuda:0 and cuda:1 tensors, asserts three fusion groups and that the
        # debug count of cached kernel specs grew by exactly one. Assumes no other test already
        # used the same kernel.
        def not_fusible(x):
            # [FUNCTION DOC] HELPER: Identity fixture that prevents fusing across it.
            return x

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Fixture multiplying each input five times and returning each
            # result through the identity function.
            x_out = x * x * x * x * x  # fusion: lambda x. x * x * x * x * x
            y_out = y * y * y * y * y
            z_out = z * z * z * z * z
            return not_fusible(x_out), not_fusible(y_out), not_fusible(z_out)

        inputs = [
            torch.randn(4, 4, dtype=torch.float),
            torch.randn(4, 4, dtype=torch.float, device='cuda:0'),
            torch.randn(4, 4, dtype=torch.float, device='cuda:1'),
        ]

        prev_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()

        # There are 3 FusionGroups. Because they have the same graph, they
        # should reuse the same KernelSpec in the KernelSpec cache.
        ge = self.checkScript(fn, inputs)
        self.assertGraphContainsExactly(
            ge.graph_for(*inputs), 'prim::FusionGroup', 3, True)
        new_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()
        # XXX: This assumes that the same kernel isn't already used by another test
        self.assertEqual(new_cache_size - prev_cache_size, 1)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    def test_nonzero_device_cuda(self):
        # [FUNCTION DOC] Traces a sigmoid-tanh expression on tensors placed on device cuda:1 and
        # asserts full fusion. Needs multiple GPUs.
        device = 'cuda:' + str(1)
        x = torch.tensor([0.4], dtype=torch.float, device=device)
        y = torch.tensor([0.7], dtype=torch.float, device=device)

        def doit(x, y):
            # [FUNCTION DOC] HELPER: Fixture computing the sigmoid of the tanh of x times x plus y,
            # plus x.
            return torch.sigmoid(torch.tanh(x * (x + y) + x))

        ge = self.checkTrace(doit, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_lstm_cuda(self):
        # [FUNCTION DOC] Gets LSTM training inputs on CUDA and runs checkScript on the LSTM cell
        # fixture, then returns.
        # KNOWN ISSUE: the function returns right after checkScript, so the fusion-group, FileCheck
        # and backward-graph assertions below it are unreachable dead code.
        inputs = get_lstm_inputs('cuda', training=True)
        module = self.checkScript(LSTMCellS, inputs)
        return
        forward_graph = module.graph_for(*inputs)
        self.assertGraphContainsExactly(
            forward_graph, 'prim::FusionGroup', 1, consider_subgraphs=True)
        self.assertTrue(len(strip_profiling_nodes(forward_graph.nodes())) == 2)
        # Everything is differentiable but TupleConstruct return
        FileCheck().check("DifferentiableGraph").check_next("TupleConstruct") \
            .check_next("return").run(str(forward_graph))

        with enable_profiling_mode_for_profiling_tests(True):
            hy, cy = module(*inputs)
            warmup_backward((hy + cy).sum())
            backward = backward_graph(module)
        self.assertAllFused(backward, except_for=("aten::t", "aten::mm",
                                                  "aten::_grad_sum_to_size"))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # By default, on Ampere or later GPUs, LSTM computes float tensors at TF32 precision.
    # We want float tensors to be computed at full precision in order to use the default precision
    @with_tf32_off
    def test_lstm_concat_cuda(self):
        # [FUNCTION DOC] Traces the concatenating LSTM cell on CUDA with TF32 turned off and checks
        # the graph has a fused concat directly followed by return.
        inputs = get_lstm_inputs('cuda')
        ge = self.checkTrace(LSTMCellC, inputs)
        graph = ge.graph_for(*inputs)
        FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_lstm_gates_permutations_cuda(self):
        # [FUNCTION DOC] For every permutation of the four terms of the LSTM gate sum, builds the
        # source text of a cell function, compiles it both with exec (eager) and as a compilation
        # unit, asserts equal outputs on CUDA inputs, and that the scripted graph has exactly one
        # fusion group. Needs CUDA.
        # lstm has gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh.
        # Test that any permutation of this will still result in one FusionGroup.
        choices = ['x.mm(w_ih.t())', 'hx.mm(w_hh.t())', 'b_ih', 'b_hh']
        template = dedent('''
        def cell(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
            gates = {} + {} + {} + {}
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            return ingate * forgetgate * cellgate * outgate
        ''')
        for permutation in permutations(choices, len(choices)):
            code = template.format(*permutation)
            scope = {}
            exec(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)

            inputs = get_lstm_inputs('cuda', training=False)
            self.assertEqual(cu.cell(*inputs), scope['cell'](*inputs))
            forward_graph = cu.cell.graph_for(*inputs)
            self.assertGraphContainsExactly(forward_graph, 'prim::FusionGroup', 1)

    # TODO: Fuser doesn't work at all when inputs require grad. Fix that
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    # By default, on Ampere or later GPUs, LSTM computes float tensors at TF32 precision.
    # We want float tensors to be computed at full precision in order to use the default precision
    @with_tf32_off
    def test_lstm_traced_cuda(self):
        # [FUNCTION DOC] Traces the LSTM cell fixture on CUDA with TF32 off and uses FileCheck to
        # assert chunk, sigmoid and tanh are absent outside a fusion group, a fusion group is
        # followed by tuple construct and return, and no second fusion group exists.
        inputs = get_lstm_inputs('cuda')
        ge = self.checkTrace(LSTMCellF, inputs)
        graph = ge.graph_for(*inputs)
        # .check_not("aten::add") don't get pulled into FusionGroup because of BailOuts
        FileCheck().check_not("Chunk").check_not("aten::sigmoid") \
            .check_not("aten::tanh").check("FusionGroup").check_next("TupleConstruct") \
            .check_next("return").check_not("FusionGroup_2").run(str(graph))

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @unittest.skip("Test is flaky, see https://github.com/pytorch/pytorch/issues/8746")
    @enable_cpu_fuser
    def test_lstm_traced_cpu(self):
        # [FUNCTION DOC] Skipped (flaky). Traces the LSTM cell on CPU and checks a fusion group
        # exists; if compilation fails with a message containing the failure text, warns and raises
        # SkipTest, otherwise re-raises.
        # KNOWN ISSUE: the warnings module is never imported (hidden by a lint suppression) and
        # FileCheck is used as a class rather than an instance, so this body would fail if run.
        inputs = get_lstm_inputs('cpu')
        try:
            ge = self.checkTrace(LSTMCellF, inputs)
            graph = ge.graph_for(*inputs)
            FileCheck.check("FusionGroup").run(str(graph))
        except RuntimeError as e:
            if 'Failed to compile' in e.args[0]:
                warnings.warn('CPU fuser test has failed! This is not a hard failure, '  # noqa: F821
                              'because the kernels sometimes trigger bugs in compilers '
                              '(most notably GCC 7.2).')
                raise unittest.SkipTest('Failed to compile') from e
            else:
                raise

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_milstm_cuda(self):
        # [FUNCTION DOC] Scripts the multiplicative-integration LSTM cell on CUDA training inputs,
        # asserts one fusion group including subgraphs, checks forward graph order with FileCheck,
        # then runs forward and warms up backward. Needs CUDA.
        inputs = get_milstm_inputs('cuda', training=True)
        module = self.checkScript(MiLSTMCell, inputs)
        forward_graph = module.graph_for(*inputs)
        self.assertGraphContainsExactly(
            forward_graph, 'prim::FusionGroup', 1, consider_subgraphs=True)
        FileCheck().check("DifferentiableGraph").check_next("TupleConstruct") \
            .check_next("return").check("FusionGroup").run(str(forward_graph))
        hy, cy = module(*inputs)
        warmup_backward((hy + cy).sum())

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, "borked on the legacy executor")
    def test_rand_cuda(self):
        # [FUNCTION DOC] Scripts a module whose method adds random noise to x squared
        # plus x, calls it twice and checks the outputs differ and lie in zero
        # (inclusive) to one (exclusive) after the deterministic part is zero, then
        # asserts full fusion. Skipped under the legacy executor.
        class M(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module with a constant device attribute and a script
            # method computing the square of x plus x plus uniform random noise like x.
            __constants__ = ['d']

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and stores a CUDA device as
                # constant d.
                super().__init__()
                self.d = torch.device('cuda')

            @torch.jit.script_method
            def create(self, x):
                # [FUNCTION DOC] HELPER: Script method returning x times x plus x plus uniform
                # random numbers shaped like x.
                return x * x + x + torch.rand_like(x)

        x = torch.zeros([3, 4, 5], dtype=torch.float, device='cuda')
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
        # [FUNCTION DOC] HELPER: Static fixture computing the relu of x plus half of y; shared with
        # the half precision test.
        return F.relu(x + .5 * y)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_relu_cuda(self):
        # [FUNCTION DOC] Traces the relu fixture on CUDA and asserts full fusion.
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(self.fn_test_relu, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_erf_cuda(self):
        # [FUNCTION DOC] Traces a function computing relu of erf minus erfc on CUDA, asserts full
        # fusion, then with grad required asserts fusion again allowing size-related nodes.
        def fn_test_erf(x):
            # [FUNCTION DOC] HELPER: Fixture computing the relu of the error function minus the
            # complementary error function of x.
            return F.relu(torch.erf(x) - torch.erfc(x))

        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        ge = self.checkTrace(fn_test_erf, (x,))
        self.assertAllFused(ge.graph_for(x))
        x.requires_grad_(True)
        ge = self.checkTrace(fn_test_erf, (x,))
        self.assertAllFused(ge.graph_for(x), except_for=("aten::size", "prim::BroadcastSizes",
                                                         "aten::_size_if_not_equal"))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, "borked on the legacy executor")
    def test_rand_broadcast_cuda(self):
        # [FUNCTION DOC] Scripts a function that multiplies by a random tensor shaped like y
        # (broadcast against x) and adds x. Asserts fusion without and with grad required, then
        # with ones input checks the broadcast random output has equal rows. Skipped under the
        # legacy executor.
        def fn_test_rand(x, y):
            # [FUNCTION DOC] HELPER: Fixture computing uniform random like y times x plus x.
            r = torch.rand_like(y)
            return r * x + x

        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')
        script_f = torch.jit.script(fn_test_rand)
        out = script_f(x, y)
        self.assertAllFused(script_f.graph_for(x, y))
        x.requires_grad_(True)
        out = script_f(x, y)
        self.assertAllFused(script_f.graph_for(x, y), except_for=("aten::size", "prim::BroadcastSizes",
                                                                  "aten::_size_if_not_equal"))
        # test that broadcasting random produces correct results
        x = torch.ones(4, 4, dtype=torch.float, device='cuda')
        y = torch.ones(4, dtype=torch.float, device='cuda')
        out = script_f(x, y)
        self.assertEqual(out[0], out[1])

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_scalar(self):
        # [FUNCTION DOC] Scripts a function with two zero-dimensional CPU tensors
        # (double the first plus the second) with the CPU fuser enabled and asserts
        # full fusion; skipped on Sandcastle.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Fixture computing two times x plus y.
            return 2 * x + y

        x = torch.tensor(0.1, dtype=torch.float, device='cpu')
        y = torch.tensor(1, dtype=torch.float, device='cpu')
        ge = self.checkScript(fn, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_small_constant_cuda(self):
        # [FUNCTION DOC] Traces a function with very small constants (around 1e-8) scaled back up,
        # checking they survive in the fused kernel and the graph is fully fused. Needs CUDA.
        def fn_test_small_constant(x, y):
            # [FUNCTION DOC] HELPER: Fixture scaling x and y by tiny constants, adding and
            # multiplying by the inverse magnitude.
            return (1e-8 * x + 5e-9 * y) * 1e8
        x = torch.randn(4, 4, dtype=torch.float, device='cuda')
        y = torch.randn(4, 4, dtype=torch.float, device='cuda')

        ge = self.checkTrace(fn_test_small_constant, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_tensor_scalar_ops_cuda(self):
        # [FUNCTION DOC] Checks that scalar constants are fused but runtime scalars are not: a
        # function adding a constant is fully fused, while a function converting a tensor argument
        # to int has zero fusion groups even including subgraphs. Needs CUDA.
        def should_fuse(x):
            # [FUNCTION DOC] HELPER: Fixture adding a constant three to x and
            # multiplying by that sum.
            z = 3.
            y = x + z
            return x * y

        # XXX: right now we only support fusing scalars if
        # they're constant (#9940)
        def should_not_fuse(x, z):
            # [FUNCTION DOC] HELPER: Fixture converting z to an integer, adding it to x and
            # multiplying by that sum.
            y = x + int(z)
            return x * y

        inputs = [torch.randn(2, 2, dtype=torch.float, device='cuda')]
        ge = self.checkScript(should_fuse, inputs)
        self.assertAllFused(ge.graph_for(*inputs))

        inputs = [
            torch.randn(2, 2, dtype=torch.float, device='cuda'),
            torch.tensor(3., dtype=torch.float, device='cuda'),
        ]
        ge = self.checkScript(should_not_fuse, inputs)
        self.assertGraphContainsExactly(
            ge.graph_for(*inputs), 'prim::FusionGroup', 0, consider_subgraphs=True)

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser CPU support for Sandcastle")
    @enable_cpu_fuser
    def test_where_and_typing(self):
        # [FUNCTION DOC] Scripts a function returning a greater-than mask and a where selection on
        # double CPU tensors with the CPU fuser enabled; asserts full fusion allowing tuple
        # construct. Skipped on Sandcastle.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Fixture returning the mask of x above y and the elementwise
            # selection of x where the mask holds, else y.
            mask = x > y
            res = torch.where(mask, x, y)
            return mask, res

        x = torch.randn(4, 4, dtype=torch.double)
        y = torch.randn(4, 4, dtype=torch.double)

        script_f = self.checkScript(f, (x, y))
        self.assertAllFused(script_f.graph_for(x, y), except_for={'prim::TupleConstruct'})

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no half support with profiling on")
    def test_grad_sum_to_size_elimination(self):
        # [FUNCTION DOC] Checks sum-to-size elimination for a broadcasting cell: scripts the cell
        # with profiling on and asserts forward fusion, then loops three times over input shape
        # combinations, rescripts, warms up backward, checks gradient shapes match inputs, picks
        # the one new backward graph and asserts it has one sum-to-size node except in the first
        # iteration, where it has none. Needs CUDA and the legacy executor.

        def my_broadcasted_cell(a, b, c):
            # [FUNCTION DOC] HELPER: Fixture adding a, b and c.
            return (a + b) + c

        s1 = torch.randn(5, 1, requires_grad=True, device='cuda')
        s2 = torch.randn(5, 5, requires_grad=True, device='cuda')

        module = self.checkScript(my_broadcasted_cell, (s1, s1, s1), profiling=ProfilingMode.PROFILING)
        forward_graph = module.graph_for(s1, s1, s1)
        self.assertAllFused(forward_graph, except_for=("aten::size", "prim::BroadcastSizes",
                                                       "aten::_size_if_not_equal"))

        old_plans = set()
        for i in range(3):
            # if we have s2, then the s1 are _grad_sum_to_size'd

            args = s2 if i < 1 else s1, s2 if i < 2 else s1, s2
            args = [a.detach_().requires_grad_() for a in args]
            # recompile, so we don't trigger bailouts
            module = self.checkScript(my_broadcasted_cell, args, profiling=ProfilingMode.PROFILING)
            res = module(s2 if i < 1 else s1, s2 if i < 2 else s1, s2)
            warmup_backward(res.sum(), args)
            grads = torch.autograd.grad(res.sum(), args)
            for inp, gr in zip(args, grads):
                self.assertEqual(inp.shape, gr.shape)
            backward = None
            # this is a workaround for the backward graphs not being
            # in order for Python 2
            for g in all_backward_graphs(module):
                if str(g) not in old_plans:
                    if backward is not None:
                        raise AssertionError("backward should be None")
                    backward = g
                    old_plans.add(str(backward))
            num_grads = 1 if i > 0 else 0
            self.assertEqual(len([n for n in backward.nodes() if n.kind() == 'aten::_grad_sum_to_size']), num_grads)


if __name__ == '__main__':
    if sys.version_info < (3, 14):
        run_tests()
