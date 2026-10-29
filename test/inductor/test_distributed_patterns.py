# Owner(s): ["oncall: pt2"]
# [MODULE DOC] Inductor/Dynamo tests of patterns used by FSDP-style distributed training, run
# without real distribution: fake all-gather and reduce-scatter via module hooks, storage resize
# to zero and back, unsafe version-counter manipulation, module backward hooks under compiled
# autograd, tensor hooks with closures, and returning nn.Parameter from compiled code. Defines a
# world size constant of 2 and helper builders.
import dataclasses
import functools

import torch
from torch import nn
from torch._dynamo import compiled_autograd
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, requires_gpu


# Fake distributed
WORLD_SIZE = 2


def init_fake_distributed(device="cpu"):
    # [FUNCTION DOC] Builds a bias-free linear module (20 to 10) that mimics FSDP with a world
    # size of 2 and returns it with a random input requiring grad of shape 2 by 20. The module
    # keeps a sharded and an unsharded weight parameter (unsharded storage is freed to zero).
    # Registers forward pre and post and backward pre and post hooks that allocate and fill the
    # unsharded weight by the fake all-gather, swap it in as the weight, then swap the sharded
    # one back and free storage; the backward post hook also reduce-scatters the gradient into
    # the sharded weight's grad. Seeds the RNG.
    @torch.no_grad
    def all_gather(t):
        # [FUNCTION DOC] HELPER: Fake all-gather: concatenates the tensor with itself once per
        # world size along the first dimension, without grad.
        return torch.cat([t] * WORLD_SIZE, 0)

    @torch.no_grad
    def reduce_scatter(t):
        # [FUNCTION DOC] HELPER: Fake reduce-scatter: returns a copy of the first shard along
        # dim 0 (size divided by world size); cloned so the output does not alias the input. No
        # grad.
        # clone since reduce_scatter input and output should not be aliases.
        return t.narrow(0, 0, t.size(0) // WORLD_SIZE).clone()

    def fw_pre_hook(mod, inp):
        # [FUNCTION DOC] HELPER: Forward pre-hook: resizes the unsharded weight storage to full
        # size, copies the all-gathered sharded weight into it through the fsdp copy op while
        # preserving the version counter, and installs it as the module weight.
        mod.unsharded_weight.untyped_storage().resize_(
            mod.unsharded_weight.nelement() * mod.unsharded_weight.element_size()
        )
        with (
            torch.no_grad(),
            torch.autograd._unsafe_preserve_version_counter(mod.unsharded_weight),
        ):
            torch.ops.fsdp.copy_(mod.unsharded_weight, all_gather(mod.sharded_weight))
        mod._parameters["weight"] = mod.unsharded_weight

    # Forward:
    #   mod.sharded_weight = local_shard (always)
    #   Before:
    #     mod.weight = local_shard
    #     mod.unsharded_weight = zero-sized allgather
    #   After:
    #     mod.weight = local_shard
    #     mod.unsharded_weight = zero-sized allgather

    def fw_post_hook(mod, inp, out):
        # [FUNCTION DOC] HELPER: Forward post-hook: reinstates the sharded weight as the module
        # weight and frees the unsharded weight storage.
        mod._parameters["weight"] = mod.sharded_weight
        mod.unsharded_weight.untyped_storage().resize_(0)

    def bw_pre_hook(mod, gO):
        # [FUNCTION DOC] HELPER: Backward pre-hook: same unshard sequence as the forward pre-
        # hook (resize storage, copy all-gathered shard preserving version counter, install as
        # weight).
        mod.unsharded_weight.untyped_storage().resize_(
            mod.unsharded_weight.nelement() * mod.unsharded_weight.element_size()
        )
        with (
            torch.no_grad(),
            torch.autograd._unsafe_preserve_version_counter(mod.unsharded_weight),
        ):
            torch.ops.fsdp.copy_(mod.unsharded_weight, all_gather(mod.sharded_weight))
        mod._parameters["weight"] = mod.unsharded_weight

    # Backward:
    #   mod.sharded_weight = local_shard (always)
    #   Before:
    #     mod.weight = local_shard
    #     mod.unsharded_weight = zero-sized allgather
    #   After:
    #     mod.weight = local_shard
    #     mod.unsharded_weight = zero-sized allgather

    def bw_post_hook(mod, gI, gO):
        # [FUNCTION DOC] HELPER: Backward post-hook: reduce-scatters the current weight
        # gradient, reinstates the sharded weight, assigns the scattered gradient to its grad,
        # and frees the unsharded storage.
        grad = mod.weight.grad
        new_grad = reduce_scatter(grad)
        mod._parameters["weight"] = mod.sharded_weight
        mod.weight.grad = new_grad
        mod.unsharded_weight.untyped_storage().resize_(0)

    torch.manual_seed(1234)
    m = nn.Linear(20, 10, bias=False, device=device)

    # Mimics eager 1st iteration
    m.sharded_weight = nn.Parameter(reduce_scatter(m.weight))
    m.unsharded_weight = nn.Parameter(all_gather(m.sharded_weight))
    m.unsharded_weight.untyped_storage().resize_(0)

    m.register_full_backward_pre_hook(bw_pre_hook)
    m.register_full_backward_hook(bw_post_hook)
    m.register_forward_pre_hook(fw_pre_hook)
    m.register_forward_hook(fw_post_hook)
    return m, torch.rand(2, 20, requires_grad=True, device=device)


def init_module_bw_hooks(allow_eager):
    # [FUNCTION DOC] Builds a 10 by 10 linear module with scalar counter tensors for pre and
    # post backward hooks and registers full backward pre and post hooks that increment the
    # counters and return transformed gradients (a sine of shifted gradient). Returns the module
    # and a random input of shape 2 by 10 requiring grad. Argument allow_eager controls whether
    # the hooks may run outside compilation. Seeds the RNG.
    def bw_pre_hook(mod, gO):
        # [FUNCTION DOC] HELPER: Backward pre-hook: asserts it is allowed (eager allowed or
        # compiling) and that the weight is 10 by 10, bumps the pre counter, and returns the
        # sine of the output gradient plus 1.2 as a tuple.
        assert allow_eager or torch._dynamo.is_compiling()  # noqa: S101
        assert mod.weight.size() == (10, 10)  # noqa: S101
        mod.hook_count_pre.add_(1)
        return (torch.sin(gO[0] + 1.2),)

    def bw_post_hook(mod, gI, gO):
        # [FUNCTION DOC] HELPER: Backward post-hook: same assertions, bumps the post counter,
        # and returns the sine of the input gradient plus 3.4 as a tuple.
        assert allow_eager or torch._dynamo.is_compiling()  # noqa: S101
        assert mod.weight.size() == (10, 10)  # noqa: S101
        mod.hook_count_post.add_(1)
        return (torch.sin(gI[0] + 3.4),)

    torch.manual_seed(1234)
    m = nn.Linear(10, 10)
    m.hook_count_pre = torch.tensor(0)
    m.hook_count_post = torch.tensor(0)
    m.register_full_backward_pre_hook(bw_pre_hook)
    m.register_full_backward_hook(bw_post_hook)
    return m, torch.rand(2, 10, requires_grad=True)


def steps(m, inp):
    # [FUNCTION DOC] HELPER: Runs four iterations of forward and a backward pass on the sum of
    # the output; returns the last output.
    for _ in range(4):
        out = m(inp)
        out.sum().backward()
    return out


class DistributedPatternTests(TestCase):
    # [CLASS DOC] TestCase with tests for the distributed-style patterns above, comparing eager
    # results to compiled ones.
    def test_intermediate_hook_with_closure(self):
        # [FUNCTION DOC] Registers a tensor hook capturing a closure variable and a dataclass
        # object; compares gradients of the eager function with the compiled (full graph) one
        # run under compiled autograd, for two different objects.
        @dataclasses.dataclass
        class CustomObj:
            # [CLASS DOC] Dataclass holding one tensor field val.
            val: torch.Tensor

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Computes a sine, registers a gradient hook adding the
            # object's value and a closure tensor, then returns the sine of the result.
            y = x.sin()
            closure_var = y + 1
            y.register_hook(lambda grad: grad + obj.val + closure_var)
            z = y.sin()
            return z

        opt = torch.compile(fn, fullgraph=True)

        obj1 = CustomObj(torch.tensor(88))
        obj2 = CustomObj(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_intermediate_hook_with_nested_closure(self):
        # [FUNCTION DOC] Like the closure test but the hooked computation is inside a nested
        # function called by the traced one; compares eager and compiled gradients for two
        # objects.
        @dataclasses.dataclass
        class CustomObj:
            # [CLASS DOC] Dataclass holding one tensor field val.
            val: torch.Tensor

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Defines and calls an inner function that registers a
            # gradient hook using the object and a closure tensor; returns its result.
            def run():
                # [FUNCTION DOC] HELPER: Computes sine, registers the gradient hook, and returns
                # the sine of the result.
                y = x.sin()
                closure_var = y + 1
                y.register_hook(lambda grad: grad + obj.val + closure_var)
                z = y.sin()
                return z

            return run()

        opt = torch.compile(fn, fullgraph=True)

        obj1 = CustomObj(torch.tensor(88))
        obj2 = CustomObj(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    @torch.no_grad()
    def _test_storage_resize_zero(self, device):
        # [FUNCTION DOC] HELPER: Compiles a function that computes sine of the input, resizes
        # the input's storage to zero, then returns cosine of the sine; checks the output and
        # that the storage size became zero. Runs without grad.
        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Computes sine, frees the input storage, returns the cosine
            # of the intermediate.
            y = torch.sin(x)
            x.untyped_storage().resize_(0)
            return torch.cos(y)

        x = torch.randn(10, device=device)
        expected = torch.cos(torch.sin(x))
        y = fn(x)
        self.assertEqual(y, expected)
        self.assertEqual(x.untyped_storage().size(), 0)

    def test_storage_resize_zero_cpu(self):
        # [FUNCTION DOC] Runs the storage resize to zero check on cpu.
        self._test_storage_resize_zero("cpu")

    @requires_gpu()
    def test_storage_resize_zero_gpu(self):
        # [FUNCTION DOC] Runs the storage resize to zero check on the GPU; requires a GPU.
        self._test_storage_resize_zero(GPU_TYPE)

    @torch.no_grad()
    def _test_storage_resize_nonzero(self, device):
        # [FUNCTION DOC] HELPER: Compiles a function that writes cosine of sine of x into an
        # output tensor whose storage has been freed, after resizing it to the input's storage
        # size; checks storage size and values. The function asserts the output storage starts
        # at size zero.
        @torch.compile(fullgraph=True)
        def fn(x, out):
            # [FUNCTION DOC] HELPER: Computes sine, asserts output storage size is zero, resizes
            # it to the input storage size and copies the cosine in.
            y = torch.sin(x)
            assert out.untyped_storage().size() == 0  # noqa: S101
            out.untyped_storage().resize_(x.untyped_storage().size())
            out.copy_(y.cos())

        x = torch.randn(10, device=device)
        out = torch.randn(10, device=device)
        expected = torch.cos(torch.sin(x))
        out.untyped_storage().resize_(0)
        fn(x, out)
        self.assertEqual(out.untyped_storage().size(), x.untyped_storage().size())
        self.assertEqual(out, expected)

    def test_storage_resize_nonzero_cpu(self):
        # [FUNCTION DOC] Runs the storage resize non-zero check on cpu.
        self._test_storage_resize_nonzero("cpu")

    @requires_gpu()
    def test_storage_resize_nonzero_gpu(self):
        # [FUNCTION DOC] Runs the storage resize non-zero check on the GPU; requires a GPU.
        self._test_storage_resize_nonzero(GPU_TYPE)

    @torch.no_grad()
    def test_unsafe_set_version_counter1(self):
        # [FUNCTION DOC] For several starting versions, compiles with a counting backend a
        # function that overwrites a tensor in place then restores its version counter via the
        # unsafe setter; checks the same object is returned, values, the version equal to the
        # original, and one compiled frame.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(w, x):
            # [FUNCTION DOC] HELPER: Overwrites w in place with the sine of x plus one, then
            # resets its version counter to the recorded one; returns w and the recorded
            # version.
            x = x.sin()
            v = w._version
            w.copy_(x + 1)
            torch._C._autograd._unsafe_set_version_counter((w,), (v,))
            return w, v

        for v in (3, 0, 1):
            w1 = torch.randn(16)
            for i in range(v):
                w1.fill_(i)  # bump w1._version
            self.assertEqual(w1._version, v)
            x1 = torch.randn(16)
            w2, v2 = fn(w1, x1)

            self.assertIs(w1, w2)
            self.assertEqual(w1, x1.sin() + 1)
            self.assertEqual(v2, v)
            self.assertEqual(w1._version, v)
            self.assertEqual(cnt.frame_count, 1)

    def test_unsafe_set_version_counter2(self):
        # [FUNCTION DOC] Compiles with inductor a function that takes the sine of w, then under
        # no-grad overwrites w and restores the version counter; checks the output, that w was
        # overwritten, and backward gives the cosine of the new value.
        @torch.compile(backend="inductor", fullgraph=True)
        def fn(w, x):
            # [FUNCTION DOC] HELPER: Takes the sine of w, then under no-grad copies x into w and
            # restores w's version counter; returns the sine.
            r = w.sin()
            with torch.no_grad():
                v = w._version
                w.copy_(x)
                torch._C._autograd._unsafe_set_version_counter((w,), (v,))
            return r

        w1 = torch.randn(1, requires_grad=True)
        x1 = torch.randn(1)
        expected_r1 = w1.detach().sin()

        r1 = fn(w1, x1)
        r1.backward()
        self.assertEqual(r1, expected_r1)
        self.assertEqual(w1, x1)
        self.assertEqual(w1.grad, x1.cos())

    @torch.no_grad()
    def test_unsafe_preserve_version_counter1(self):
        # [FUNCTION DOC] Same idea as the first version-counter test but using the preserve-
        # version-counter context manager with the eager backend; checks the version is
        # unchanged and values updated.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(w, x):
            # [FUNCTION DOC] HELPER: Within the preserve-version-counter context, copies sine of
            # x plus one into w; returns w.
            x = x.sin()
            with torch.autograd._unsafe_preserve_version_counter(w):
                w.copy_(x + 1)
            return w

        w1 = torch.randn(16).fill_(0).fill_(1)
        x1 = torch.randn(16)
        v1 = w1._version
        w2 = fn(w1, x1)
        v2 = w1._version

        self.assertIs(w1, w2)
        self.assertEqual(w1, x1.sin() + 1)
        self.assertEqual(v1, v2)

    def test_unsafe_preserve_version_counter2(self):
        # [FUNCTION DOC] Inductor-compiled version of the preserve-version-counter check
        # combined with autograd, comparing output, overwritten value and gradient.
        @torch.compile(backend="inductor", fullgraph=True)
        def fn(w, x):
            # [FUNCTION DOC] HELPER: Takes the sine of w, then under no-grad and the preserve-
            # version-counter context copies x into w; returns the sine.
            r = w.sin()
            with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(w):
                w.copy_(x)
            return r

        w1 = torch.randn(1, requires_grad=True)
        x1 = torch.randn(1)
        expected_r1 = w1.detach().sin()

        r1 = fn(w1, x1)
        r1.backward()
        self.assertEqual(r1, expected_r1)
        self.assertEqual(w1, x1)
        self.assertEqual(w1.grad, x1.cos())

    def test_module_backward_hooks_eager(self):
        # [FUNCTION DOC] Compares eager module backward hooks to a compiled module under
        # compiled autograd with counting backends; checks hook counters, outputs, gradients,
        # one forward frame with 5 ops, two backward frames and 111 ops in the backward graphs.
        m1, inp1 = init_module_bw_hooks(True)
        out1 = steps(m1, inp1)

        m2, inp2 = init_module_bw_hooks(False)
        fw_cnt = CompileCounter()
        bw_cnt = CompileCounter()
        with compiled_autograd._enable(torch.compile(backend=bw_cnt, fullgraph=True)):
            m2 = torch.compile(m2, backend=fw_cnt, fullgraph=True)
            out2 = steps(m2, inp2)

        self.assertEqual(m1.hook_count_pre, m2.hook_count_pre)
        self.assertEqual(m1.hook_count_post, m2.hook_count_post)
        self.assertEqual(out1, out2)
        self.assertEqual(inp1.grad, inp2.grad)
        self.assertEqual(m1.weight.grad, m2.weight.grad)
        self.assertEqual(m1.bias.grad, m2.bias.grad)

        self.assertEqual(fw_cnt.frame_count, 1)
        self.assertEqual(fw_cnt.op_count, 5)
        self.assertEqual(bw_cnt.frame_count, 2)  # grad=None and grad!=None
        self.assertEqual(
            bw_cnt.op_count, 111
        )  # Number of ops in the Dynamo-produced graphs

    def test_module_backward_hooks_aot(self):
        # [FUNCTION DOC] Compares eager hooks vs the aot_eager compiled module with an identity
        # compiled-autograd compiler; checks counters, outputs and gradients.
        m1, inp1 = init_module_bw_hooks(True)
        out1 = steps(m1, inp1)

        m2, inp2 = init_module_bw_hooks(True)
        m2 = torch.compile(m2, backend="aot_eager", fullgraph=True)
        with compiled_autograd._enable(lambda gm: gm):
            out2 = steps(m2, inp2)

        self.assertEqual(m1.hook_count_pre, m2.hook_count_pre)
        self.assertEqual(m1.hook_count_post, m2.hook_count_post)
        self.assertEqual(out1, out2)
        self.assertEqual(inp1.grad, inp2.grad)
        self.assertEqual(m1.weight.grad, m2.weight.grad)
        self.assertEqual(m1.bias.grad, m2.bias.grad)

    def test_module_backward_hooks_inductor(self):
        # [FUNCTION DOC] Compares eager hooks vs the default inductor compile of the module with
        # inductor compiled autograd; checks counters, outputs and gradients.
        m1, inp1 = init_module_bw_hooks(True)
        out1 = steps(m1, inp1)

        m2, inp2 = init_module_bw_hooks(False)
        m2 = torch.compile(m2, fullgraph=True)
        with compiled_autograd._enable(torch.compile(fullgraph=True)):
            out2 = steps(m2, inp2)

        self.assertEqual(m1.hook_count_pre, m2.hook_count_pre)
        self.assertEqual(m1.hook_count_post, m2.hook_count_post)
        self.assertEqual(out1, out2)
        self.assertEqual(inp1.grad, inp2.grad)
        self.assertEqual(m1.weight.grad, m2.weight.grad)
        self.assertEqual(m1.bias.grad, m2.bias.grad)

    def test_module_backward_hooks_multi_layers(self):
        # [FUNCTION DOC] Two stacked hooked modules in a sequential container, eager vs compiled
        # with compiled autograd; checks counters of both modules, output, and all gradients.
        a1, inp1 = init_module_bw_hooks(True)
        b1, _ = init_module_bw_hooks(True)
        out1 = steps(torch.nn.Sequential(a1, b1), inp1)

        a2, inp2 = init_module_bw_hooks(False)
        b2, _ = init_module_bw_hooks(False)
        with compiled_autograd._enable(torch.compile(fullgraph=True)):
            out2 = steps(
                torch.compile(torch.nn.Sequential(a2, b2), fullgraph=True), inp2
            )

        self.assertEqual(a1.hook_count_pre, a2.hook_count_pre)
        self.assertEqual(a1.hook_count_post, a2.hook_count_post)
        self.assertEqual(b1.hook_count_pre, b2.hook_count_pre)
        self.assertEqual(b1.hook_count_post, b2.hook_count_post)
        self.assertEqual(out1, out2)
        self.assertEqual(inp1.grad, inp2.grad)
        self.assertEqual(a1.weight.grad, a2.weight.grad)
        self.assertEqual(a1.bias.grad, a2.bias.grad)
        self.assertEqual(b1.weight.grad, b2.weight.grad)
        self.assertEqual(b1.bias.grad, b2.bias.grad)

    # TODO(jansel): support bw hooks with graph break

    def _assert_same_grad(self, a, b):
        # [FUNCTION DOC] HELPER: Asserts two tensors have the same type, values, gradients and
        # requires-grad flag.
        self.assertEqual(type(a), type(b))
        self.assertEqual(a, b)
        self.assertEqual(a.grad, b.grad)
        self.assertEqual(a.requires_grad, b.requires_grad)

    def test_nn_param_return1(self):
        # [FUNCTION DOC] Checks a compiled function that creates a parameter from its input and
        # returns it with its sine behaves like eager, including gradients.
        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps the input in a parameter and returns it with its
            # sine.
            p = torch.nn.Parameter(x)
            return p, p.sin()

        opt = torch.compile(fn, fullgraph=True)
        x1 = torch.randn(16)
        x2 = x1.clone()

        p1, r1 = fn(x1)
        r1.sum().backward()
        p2, r2 = opt(x2)
        r2.sum().backward()
        self._assert_same_grad(r1, r2)
        self._assert_same_grad(p1, p2)

    def test_nn_param_return2(self):
        # [FUNCTION DOC] Checks returning a parameter created with no grad requirement alongside
        # an offset of the input matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps the input in a non-grad parameter; returns it and the
            # input plus one.
            p = torch.nn.Parameter(x, requires_grad=False)
            return p, x + 1

        opt = torch.compile(fn, fullgraph=True)
        x1 = torch.randn(16)
        x2 = x1.clone()

        p1, r1 = fn(x1)
        p2, r2 = opt(x2)
        self._assert_same_grad(r1, r2)
        self._assert_same_grad(p1, p2)

    @torch._dynamo.config.patch("graph_break_on_nn_param_ctor", False)
    def test_nn_param_return3(self):
        # [FUNCTION DOC] With the graph-break-on-parameter-constructor option off, checks
        # returning a parameter built from an offset input plus its sine matches eager including
        # gradients.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a parameter from the input plus 123 and returns it
            # with its sine.
            p = torch.nn.Parameter(x + 123)
            return p, p.sin()

        opt = torch.compile(fn, fullgraph=True)
        x1 = torch.randn(16)
        x2 = x1.clone()

        p1, r1 = fn(x1)
        r1.sum().backward()
        p2, r2 = opt(x2)
        r2.sum().backward()
        self._assert_same_grad(r1, r2)
        self._assert_same_grad(p1, p2)

    @torch._dynamo.config.patch("graph_break_on_nn_param_ctor", False)
    def test_nn_param_return4(self):
        # [FUNCTION DOC] With the graph-break-on-parameter-constructor option off, checks
        # returning a non-grad parameter built from an offset input matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a non-grad parameter from the input plus 123;
            # returns it and the input plus one.
            p = torch.nn.Parameter(x + 123, requires_grad=False)
            return p, x + 1

        opt = torch.compile(fn, fullgraph=True)
        x1 = torch.randn(16)
        x2 = x1.clone()

        p1, r1 = fn(x1)
        p2, r2 = opt(x2)
        self._assert_same_grad(r1, r2)
        self._assert_same_grad(p1, p2)

    @torch._functorch.config.patch(recompute_views=True)
    def test_fake_distributed_aot_eager(self):
        # [FUNCTION DOC] Runs the fake distributed module eagerly and compiled with the
        # aot_eager backend under compiled autograd (recompute views on); compares weights,
        # input and output including grads, and expects two backward frames (grad none and not
        # none).
        m1, inp1 = init_fake_distributed()
        out1 = steps(m1, inp1)

        m2, inp2 = init_fake_distributed()
        m2 = torch.compile(m2, backend="aot_eager", fullgraph=True)
        bw_cnt = CompileCounter()
        with compiled_autograd._enable(torch.compile(backend=bw_cnt, fullgraph=True)):
            out2 = steps(m2, inp2)

        self._assert_same_grad(m1.weight, m2.weight)
        self._assert_same_grad(inp1, inp2)
        self._assert_same_grad(out1, out2)
        # Recompile on grad==None/grad!=None
        self.assertEqual(bw_cnt.frame_count, 2)

    @requires_gpu()
    @torch._functorch.config.patch(recompute_views=True)
    def test_fake_distributed_inductor(self):
        # [FUNCTION DOC] Same as the aot_eager version but on the GPU with default inductor for
        # forward and compiled autograd; requires a GPU and compares weight, input and output.
        m1, inp1 = init_fake_distributed(GPU_TYPE)
        out1 = steps(m1, inp1)

        m2, inp2 = init_fake_distributed(GPU_TYPE)
        m2 = torch.compile(m2, fullgraph=True)
        with compiled_autograd._enable(torch.compile(fullgraph=True)):
            out2 = steps(m2, inp2)

        self._assert_same_grad(m1.weight, m2.weight)
        self._assert_same_grad(inp1, inp2)
        self._assert_same_grad(out1, out2)


if __name__ == "__main__":
    if HAS_CPU and not IS_MACOS:
        run_tests(needs="filelock")
