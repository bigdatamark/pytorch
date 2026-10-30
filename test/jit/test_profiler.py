# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for the profiling executor and the tensor expression fuser as seen through
# the last executed optimized graph: which ops are fused or left out, specialization of types and
# backward graphs, fallback graphs, tensor constants and fusion strategy settings. Import-time
# side effect: appends the parent test directory to the module search path. Running the file
# directly raises an error pointing at test_jit.

import os
import sys

import torch
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    skipIfTorchDynamo,
)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import FileCheck, JitTestCase, warmup_backward


@skipIfTorchDynamo()
class TestProfiler(JitTestCase):
    # [CLASS DOC] Test case (skipped under TorchDynamo) that turns on the profiling executor and
    # the tensor expression fuser (with CPU fusion, reductions on, LLVM not required, fusion group
    # inlining off) and sets the default dtype to double for each test. Previous values of these
    # global settings are saved on the instance and restored in tearDown.
    def setUp(self):
        # [FUNCTION DOC] Deliberately does not call the base class setUp, which installs hooks
        # said to crash at process exit. Records the torch function mode stack length and state,
        # then saves and overrides the global JIT settings listed in the class doc, including
        # autodiff subgraph inlining off and double default dtype.
        # Don't call super().setUp() — JitTestCase.setUp installs JIT emit
        # hooks that cause segfaults during process cleanup. Record state
        # baselines that tearDown checks for.
        self._prev_torch_function_mode_stack_len = torch._C._len_torch_function_stack()
        self._prev_torch_function_state = torch._C._get_torch_function_state()
        self.prev_exec = torch._C._jit_set_profiling_executor(True)
        self.prev_profiling = torch._C._get_graph_executor_optimize(True)
        self.inline_autodiff = torch._C._debug_set_autodiff_subgraph_inlining(False)
        self.texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
        self.can_fuse_on_cpu = torch._C._jit_can_fuse_on_cpu()
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        self.default_dtype = torch.get_default_dtype()
        self.old_reduction_enabled = torch._C._jit_set_texpr_reductions_enabled(True)
        torch.set_default_dtype(torch.double)
        self.old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        self.old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)

    def tearDown(self):
        # [FUNCTION DOC] Restores every global setting saved by setUp: executors, fuser flags,
        # autodiff inlining, default dtype, reductions, fusion group inlining and LLVM
        # requirement. Does not call the base class tearDown.
        # KNOWN ISSUE: the saved torch function stack length and state are recorded in setUp but
        # never checked here.
        torch._C._jit_set_profiling_executor(self.prev_exec)
        torch._C._get_graph_executor_optimize(self.prev_profiling)
        torch._C._debug_set_autodiff_subgraph_inlining(self.inline_autodiff)
        torch._C._jit_set_texpr_fuser_enabled(self.texpr_fuser_state)
        torch._C._jit_override_can_fuse_on_cpu(self.can_fuse_on_cpu)
        torch.set_default_dtype(self.default_dtype)
        torch._C._jit_set_texpr_reductions_enabled(self.old_reduction_enabled)
        torch._C._debug_set_fusion_group_inlining(self.old_fusion_inlining)
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_te_must_use_llvm_cpu)

    def test_tensor_type_not_determined_by_inputs(self):
        # [FUNCTION DOC] Checks that fusion does not assume a result type from input types: a
        # scripted function whose third argument becomes a float scalar tensor after warmup must
        # keep the item and add ops between two fusion groups; and a sum whose dtype argument is
        # computed at run time must stay outside the fusion groups.
        @torch.jit.script
        def scalar_type_input(x, y, z):
            # [FUNCTION DOC] HELPER: Scripted function adding two tensors, the constant four, and
            # the scalar extracted from a third tensor.
            return x + y + 4 + z.item()

        x = torch.tensor([2, 2])
        scalar_type_input(x, x, torch.tensor(1))
        scalar_type_input(x, x, torch.tensor(1))
        scalar_type_input(x, x, torch.tensor(1.0))
        g = torch.jit.last_executed_optimized_graph()

        # item & add should not get pulled into the fusion group -
        # we expect to see Fusion Group (item / add) Fusion Group in ir dump
        FileCheck().check("TensorExpr").check("Scalar = aten::item").check_next(
            "Tensor = aten::add"
        ).check("TensorExpr").run(g)

        @torch.jit.script
        def non_const_dtype(x, y, cond: bool):
            # [FUNCTION DOC] HELPER: Scripted function that picks a dtype from a boolean flag and
            # returns the sum of an expression with that dtype.
            dtype = torch.int16 if cond else torch.int32
            return (x + y + 3).sum(dtype=dtype)

        non_const_dtype(x, x, True)
        non_const_dtype(x, x, True)
        g = torch.jit.last_executed_optimized_graph()
        # because dtype is non-const, sum should not get pulled into the Fusion Group
        FileCheck().check("TensorExpr").check("TensorExpr").check_not("aten::sum").run(
            g
        )

    def test_specialize_backward(self):
        # [FUNCTION DOC] Checks the specialized backward graph of a fused function. With
        # same-shape inputs the true block of the guarding branch has exactly one nested branch
        # guarding a fusion group and no gradient sum-to-size op. With a broadcast input the
        # gradient sum-to-size op is expected to remain.
        def test_fuse(a, b):
            # [FUNCTION DOC] HELPER: Multiplies two inputs and multiplies the result by the second
            # again; the function caching flag is disabled on it by the test.
            c = a * b
            d = c * b
            return d

        test_fuse.__disable_jit_function_caching__ = True

        scripted_f = torch.jit.script(test_fuse)
        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        scripted_f(x, y)
        b = scripted_f(x, y)
        warmup_backward(b)
        g = torch.jit.last_executed_optimized_graph()
        # Backward has an if node guarding specializations,
        # within the if node true block there is only one if node
        # that guards a tensorexpr group
        optimized_block = next(g.findNode("prim::If").blocks())
        if_nodes = list(optimized_block.findAllNodes("prim::If"))

        self.assertEqual(len(if_nodes), 1)
        FileCheck().check("Group[Subgraph").run(str(if_nodes[0]))
        # no broadcasts occurred, sum_to_size have been specialized out
        self.assertIsNone(optimized_block.findNode("aten::_grad_sum_to_size"))

        broadcast_f = torch.jit.script(test_fuse)
        x = torch.ones([2, 2], requires_grad=True)
        y = torch.ones([1], requires_grad=True)
        broadcast_f(x, y)
        b = broadcast_f(x, y)
        b.backward(torch.ones([2, 2], dtype=torch.float), retain_graph=True)
        b.backward(torch.ones([2, 2], dtype=torch.float))
        # warmup_backward(b, torch.ones([2, 2], dtype=torch.float))
        g = torch.jit.last_executed_optimized_graph()
        optimized_block = next(g.findNode("prim::If").blocks())
        # broadcasts occurred, currently expect to see aten::_grad_sum_to_size
        self.assertIsNotNone(optimized_block.findNode("aten::_grad_sum_to_size"))

    def test_specialized_types(self):
        # [FUNCTION DOC] Runs a scripted function three times and checks that tensor types remain
        # specialized (double) on type check outputs and fusion outputs while the branch output is
        # a generic tensor.
        @torch.jit.script
        def test_fuse(a, b):
            # [FUNCTION DOC] HELPER: Scripted function: two chained multiplications.
            c = a * b
            d = c * b
            return d

        x = torch.tensor([0.5])
        for _ in range(3):
            test_fuse(x, x)

        g = torch.jit.last_executed_optimized_graph()
        # Types should remain specialized for typecheck outputs & fusion outputs
        FileCheck().check("Double(").check_same("prim::TypeCheck").check_same(
            "\n"
        ).check("Double").check_same("TensorExpr").run(g)

        # other outputs should not be specialized
        FileCheck().check("Tensor = prim::If").run(g)

    def test_aliasing_merge(self):
        # [FUNCTION DOC] Checks that a function with an in-place add in the middle yields two type
        # checks and that the in-place op sits between two fusion groups.
        @torch.jit.script
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Scripted function with multiplications, an in-place add on an
            # intermediate, another multiplication, and a final sum.
            c = a * b
            d = c * b
            d.add_(b)
            e = d * b
            return d + e

        x = torch.ones(1)
        y = torch.ones(1)
        foo(x, y)
        b = foo(x, y)  # noqa: F841
        g = torch.jit.last_executed_optimized_graph()
        self.assertEqual(len(list(g.findAllNodes("prim::TypeCheck"))), 2)
        FileCheck().check("TensorExpr").check("aten::add_").check("TensorExpr").run(g)

    def test_use_not_profiled(self):
        # [FUNCTION DOC] Checks that when a use of the first input sits in a branch never taken
        # during profiling, all adds are still fused (no add outside the branch) and the scripted
        # result matches eager.
        def foo(t1, t2, t3, t4, t: float):
            # [FUNCTION DOC] HELPER: Sums four tensors and, if a float argument exceeds one half,
            # returns the first tensor plus one; otherwise returns the sum.
            h = t1 + t2 + t3 + t4
            if t > 0.5:
                # Putting a use of t1 in a never-executed conditional prevents
                return t1 + 1
            return h

        t = torch.rand(8, dtype=torch.float)

        foo_script = torch.jit.script(foo)
        for _ in range(torch._C._jit_get_num_profiled_runs() + 1):
            foo_script(t, t, t, t, 0.1)

        self.assertEqual(foo(t, t, t, t, 0.1), foo_script(t, t, t, t, 0.1))
        g = torch.jit.last_executed_optimized_graph()
        # all adds fused
        FileCheck().check("graph").check_not("aten::add").check("prim::If").run(g)

    def test_not_fusing_scalar_ops(self):
        # [FUNCTION DOC] Checks that arithmetic on plain ints is not fused: the optimized graph
        # has no tensor expression group.
        @torch.jit.script
        def foo(x: int, y: int):
            # [FUNCTION DOC] HELPER: Scripted function summing two ints and several constants.
            return x + y + 2 + 4 + 5 + 6

        foo(1, 2)
        foo(2, 3)
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check_not("TensorExpr").run(g)

    def test_not_optimizing_property(self):
        # [FUNCTION DOC] Checks that a size query returned from a function stays in the optimized
        # graph, and the scripted result matches eager for a different input shape.
        @torch.jit.script
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Scripted function returning a sum with int constants together
            # with the size of the first input.
            return x + y + 1 + 2 + 3, x.size()

        x = torch.ones(1)
        foo(x, x)
        foo(x, x)
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check("aten::size").run(g)
        x = torch.ones([2, 3, 5])
        self.assertEqual(foo(x, x), (x + x + 1 + 2 + 3, x.size()))

    def test_fallback_graph_not_specialized(self):
        # [FUNCTION DOC] Checks that the fallback path in the optimized graph is called as a
        # function whose result is unpacked from a tuple, rather than a specialized copy.
        @torch.jit.script
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Scripted function: a chain of multiplications, then a sum.
            c = a * b
            d = c * b
            e = d * b
            return d + e

        x = torch.ones(1)
        y = torch.ones(1)
        foo(x, y)
        foo(x, y)
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check("CallFunction").check_next("Tensor = prim::TupleUnpack").run(
            g
        )

    def test_autograd_fallback_graph(self):
        # [FUNCTION DOC] Runs a scripted function with grad-requiring inputs, then two backward
        # passes, and checks the last executed graph calls a fallback function directly followed
        # by a call to a function.
        @torch.jit.script
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Scripted function: a chain of multiplications, then a sum.
            c = a * b
            d = c * b
            e = d * b
            return d + e

        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        foo(x, y)
        b = foo(x, y)
        b.backward(torch.ones([1], dtype=torch.float), retain_graph=True)
        b.backward(torch.ones([1], dtype=torch.float))

        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check("fallback_function").check_next("CallFunction").run(g)

    def test_tensor_constant(self):
        # [FUNCTION DOC] Checks results match eager when a tensor constant is created in the
        # function and that the optimized graph keeps exactly two add ops.
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Returns the sum of both inputs and a constant tensor.
            return a + b + torch.tensor([2])

        x = torch.ones(1, requires_grad=False)
        foo_script = torch.jit.script(foo)
        foo_script(x, x)
        foo_script(x, x)

        self.assertEqual(foo_script(x, x), foo(x, x))
        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check_count("aten::add", 2, exactly=True).run(g)

    def test_local_fusion_strategy(self):
        # [FUNCTION DOC] Sets the fusion strategy to one static specialization, runs three times,
        # then to ten static specializations and runs ten shapes twice each, and checks the final
        # graph contains exactly two tensor expression groups. Side effect: changes the global
        # fusion strategy and does not restore it.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function adding its input to itself twice.
            return x + x + x

        torch.jit.set_fusion_strategy([("STATIC", 1)])
        for _ in range(3):
            foo(torch.rand([10]))

        torch.jit.set_fusion_strategy([("STATIC", 10)])

        for i in range(10):
            foo(torch.rand([i]))
            foo(torch.rand([i]))

        g = torch.jit.last_executed_optimized_graph()
        FileCheck().check_count(":TensorExprGroup", 2, exactly=True).run(g)

    def test_invalid_fusion_strategy_raises_value_error(self):
        # [FUNCTION DOC] Checks that setting a fusion strategy containing an unknown kind raises
        # ValueError whose message mentions the allowed kinds and the bad one.
        with self.assertRaisesRegex(
            ValueError,
            r"FusionBehavior.*(STATIC|DYNAMIC).*got:\s*COMPLEX",
        ):
            torch.jit.set_fusion_strategy([("STATIC", 2), ("COMPLEX", 3)])

    def test_iterative_fusion(self):
        # [FUNCTION DOC] Checks that a function with an in-place op in the middle ends up with two
        # tensor expression groups after warmup, because fusing continues
        # past the first fusion point.
        @torch.jit.script
        def foo(a, b, c, d):
            # [FUNCTION DOC] HELPER: Scripted function with adds, an in-place add on the second
            # input, and a final add; returns two results.
            a = a + b
            b.add_(3)
            c = c + b + d
            a = a + 1
            return a, c

        x = torch.ones(1, requires_grad=False)
        foo(x, x, x, x)
        foo(x, x, x, x)

        # when we iterate through the block, we will start
        # by fusing a = a + b with a = a + 1
        # if we were to continue iteration from that fusion point,
        # would miss the fusion opportunity of c = c + d + b

        g = torch.jit.last_executed_optimized_graph()
        self.assertEqual(len(list(g.findAllNodes("prim::TensorExprGroup"))), 2)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
