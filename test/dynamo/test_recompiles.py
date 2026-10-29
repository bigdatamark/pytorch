# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that Dynamo recompiles the expected number of times under automatic dynamic
# shapes, aliasing guards, parameter shape settings, autocast state, module train/eval mode,
# recompile limits and unused-object arguments. One TestCase class using compile counters; runs
# the Dynamo test runner when executed as a script.
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import config as dc


class RecompileTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Collection of recompilation-count tests. Each test builds a CompileCounter
    # backend, calls a compiled function with varying inputs or global state, and asserts the
    # frame and op counts.
    def test_automatic_dynamic_reduce_recompiles(self):
        # [FUNCTION DOC] Compares frame/op counts of a multiply over tensors of growing size
        # with automatic dynamic shapes off (5 frames), via dynamic=False (5), on (2),
        # dynamic=None (2), and dynamic=True (1). Resets Dynamo between scenarios.
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Function under test: elementwise product.
            return x * y

        def run_foo_6_times_and_count_recompiles(dynamic=None):
            # [FUNCTION DOC] HELPER: Compiles foo with a counter backend and the given dynamic
            # setting, runs it with sizes 2, 3, 4, 4, 5, 5, 6, and returns the counter.
            cnt = torch._dynamo.testing.CompileCounter()

            x = torch.randn([2])
            y = torch.randn([2])
            opt = torch.compile(foo, backend=cnt, dynamic=dynamic)
            opt(x, y)
            x = torch.randn([3])
            y = torch.randn([3])
            opt(x, y)
            x = torch.randn([4])
            y = torch.randn([4])
            opt(x, y)
            opt(x, y)
            x = torch.randn([5])
            y = torch.randn([5])
            opt(x, y)
            opt(x, y)
            x = torch.randn([6])
            y = torch.randn([6])
            opt(x, y)

            return cnt

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            # [FUNCTION DOC] HELPER: Runs the shared scenario with automatic dynamic shapes
            # patched off.
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            # [FUNCTION DOC] HELPER: Runs the shared scenario with automatic dynamic shapes
            # patched on.
            return run_foo_6_times_and_count_recompiles()

        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        torch._dynamo.reset()
        without = run_foo_6_times_and_count_recompiles(dynamic=False)
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        torch._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 2)
        self.assertEqual(with_automatic.op_count, 2)
        torch._dynamo.reset()
        with_automatic = run_foo_6_times_and_count_recompiles(dynamic=None)
        self.assertEqual(with_automatic.frame_count, 2)
        self.assertEqual(with_automatic.op_count, 2)
        torch._dynamo.reset()
        with_dynamic = run_foo_6_times_and_count_recompiles(dynamic=True)
        self.assertEqual(with_dynamic.frame_count, 1)
        self.assertEqual(with_dynamic.op_count, 1)

    @patch.object(torch._dynamo.config, "assume_static_by_default", True)
    def test_recompiles_true_false_flop(self):
        # [FUNCTION DOC] Counts recompiles for a fullgraph function that branches on a Python
        # bool: without automatic dynamic shapes 5 frames, with it 3 frames. Resets Dynamo
        # between runs.
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Function under test: returns y times 2 if x is truthy, else
            # y times y.
            if x:
                return y * 2
            else:
                return y * y

        def run_foo_6_times_and_count_recompiles():
            # [FUNCTION DOC] HELPER: Runs the compiled branching function for (True, size 2),
            # (False, 2), (True, 3), (True, 4), (True, 5) and returns the counter.
            cnt = torch._dynamo.testing.CompileCounter()

            opt = torch.compile(foo, backend=cnt, fullgraph=True)

            x = True
            y = torch.randn([2])
            opt(x, y)
            x = False
            y = torch.randn([2])
            opt(x, y)
            x = True
            y = torch.randn([3])
            opt(x, y)
            x = True
            y = torch.randn([4])
            opt(x, y)
            x = True
            y = torch.randn([5])
            opt(x, y)

            return cnt

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            # [FUNCTION DOC] HELPER: Runs the scenario with automatic dynamic shapes patched
            # off.
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            # [FUNCTION DOC] HELPER: Runs the scenario with automatic dynamic shapes patched on.
            return run_foo_6_times_and_count_recompiles()

        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        torch._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 3)
        self.assertEqual(with_automatic.op_count, 3)

    def test_automatic_dynamic_tensor_scalar_change(self):
        # [FUNCTION DOC] Counts recompiles when the second argument alternates between tensor
        # and Python int while the size grows: 5 frames without automatic dynamic shapes, 3 with
        # it.
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Function under test: elementwise product.
            return x * y

        def run_foo_6_times_and_count_recompiles_swap_types():
            # [FUNCTION DOC] HELPER: Runs the compiled product over tensor and int second
            # arguments of varying size, and returns the counter.
            cnt = torch._dynamo.testing.CompileCounter()

            x = torch.randn([2])
            y = torch.randn([2])
            opt = torch.compile(foo, backend=cnt)
            opt(x, y)
            x = torch.randn([3])
            y = 3
            opt(x, y)
            x = torch.randn([4])
            y = torch.randn([4])
            opt(x, y)
            opt(x, y)
            x = torch.randn([5])
            y = 4
            opt(x, y)
            opt(x, y)
            x = torch.randn([6])
            y = torch.randn([6])
            opt(x, y)

            return cnt

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            # [FUNCTION DOC] HELPER: Runs the scenario with automatic dynamic shapes patched
            # off.
            return run_foo_6_times_and_count_recompiles_swap_types()

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            # [FUNCTION DOC] HELPER: Runs the scenario with automatic dynamic shapes patched on.
            return run_foo_6_times_and_count_recompiles_swap_types()

        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        torch._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 3)
        self.assertEqual(with_automatic.op_count, 3)

    def test_aliasing_guard_failures(self):
        # [FUNCTION DOC] Checks alias-aware guards: a fullgraph function that mutates its first
        # arg with the second compiles once; permuting distinct tensors does not recompile;
        # passing the same tensor as first and third args recompiles (frame count 2).
        # KNOWN ISSUE: the variable named eager_result in the last case is produced by the
        # compiled function, not the eager one, so the equality check compares compiled with
        # compiled.
        def foo(a, b, c):
            # [FUNCTION DOC] HELPER: Function under test: adds b into a in place and returns c
            # plus 1.
            a.add_(b)
            return c + 1

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_foo = torch.compile(foo, backend=cnt, fullgraph=True)

        x = torch.randn([3])
        y = torch.randn([3])
        z = torch.randn([3])
        cmp_result = compiled_foo(
            x.detach().clone(), y.detach().clone(), z.detach().clone()
        )
        eager_result = foo(x.detach().clone(), y.detach().clone(), z.detach().clone())
        self.assertEqual(cmp_result, eager_result)
        self.assertEqual(cnt.frame_count, 1)

        cmp_result = compiled_foo(
            z.detach().clone(), y.detach().clone(), x.detach().clone()
        )
        eager_result = foo(z.detach().clone(), y.detach().clone(), x.detach().clone())
        self.assertEqual(cmp_result, eager_result)
        # No recompile, alias preserved
        self.assertEqual(cnt.frame_count, 1)

        x_clone = x.detach().clone()
        cmp_result = compiled_foo(x_clone, y.detach().clone(), x_clone)
        x_clone = x.detach().clone()
        eager_result = compiled_foo(x_clone, y.detach().clone(), x_clone)
        self.assertEqual(cmp_result, eager_result)
        # Recompile, alias changed
        self.assertEqual(cnt.frame_count, 2)

    def test_object_alias_relation_guards_without_lambda(self):
        # [FUNCTION DOC] With the lambda-based object aliasing guard disabled via config, a
        # function that puts two arguments in a set and branches on its size gives 1 for the
        # same object twice and minus 1 for distinct objects, with 2 frames. Resets Dynamo at
        # the end.
        class Box:
            # [CLASS DOC] HELPER: Empty plain class used as an object whose identity is
            # compared.
            pass

        def foo(box_a, box_b, t):
            # [FUNCTION DOC] HELPER: Function under test: returns t plus 1 if both boxes are the
            # same object, else t minus 1.
            entries = {box_a, box_b}
            if len(entries) == 1:
                return t + 1
            return t - 1

        cnt = torch._dynamo.testing.CompileCounter()
        x = torch.tensor(0)

        with dc.patch(use_lamba_guard_for_object_aliasing=False):
            compiled = torch.compile(foo, backend=cnt, fullgraph=True)

            shared = Box()
            res_alias = compiled(shared, shared, x)
            self.assertEqual(res_alias.item(), 1)

            res_unique = compiled(Box(), Box(), x)
            self.assertEqual(res_unique.item(), -1)
            self.assertEqual(cnt.frame_count, 2)

        torch._dynamo.reset()

    def test_aliasing_guard_failures_with_globals(self):
        # [FUNCTION DOC] Alias guards with global tensors: a function adds a global into its
        # argument and returns another global plus 1. The first call compiles; calling with the
        # global itself as the argument (alias) recompiles, giving 2 frames.
        # KNOWN ISSUE: the variable named eager_result in the second case is produced by the
        # compiled function, not the eager one.
        g1 = torch.randn([3])
        g2 = torch.randn([3])

        def foo(a):
            # [FUNCTION DOC] HELPER: Function under test: mutates the argument in place by
            # adding g1, returns g2 plus 1.
            a.add_(g1)
            return g2 + 1

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_foo = torch.compile(foo, backend=cnt, fullgraph=True)

        z = torch.randn([3])
        cmp_result = compiled_foo(z.detach().clone())
        eager_result = foo(z.detach().clone())
        self.assertEqual(cmp_result, eager_result)
        self.assertEqual(cnt.frame_count, 1)

        g1 = g1.detach().clone()
        cmp_result = compiled_foo(g1)
        g1 = g1.detach().clone()
        eager_result = compiled_foo(g1)
        self.assertEqual(cmp_result, eager_result)
        # Recompile, alias changed
        self.assertEqual(cnt.frame_count, 2)

    def test_dynamic_shape_parameter_recompile(self):
        # [FUNCTION DOC] Counts recompiles of a matrix multiply by a global Parameter for
        # Parameter inputs of varying rows under combinations of the
        # force_parameter_static_shapes and automatic_dynamic_shapes configs: 4, 4, 4 and 2
        # frames respectively (only dynamic parameter shapes plus automatic dynamic gives 2).
        # Test the matrix multiplication with Parameters.
        # Without the config assume_parameters_shapes_static_by_default,
        # the torch.nn.Parameter shapes are assumed to be static which leads to recompilation

        w = torch.nn.Parameter(torch.randn(3, 2))

        def foo(x):
            # [FUNCTION DOC] HELPER: Function under test: matrix product of x with the global
            # parameter.
            return x @ w

        def run_foo_6_times_and_count_recompiles():
            # [FUNCTION DOC] HELPER: Compiles foo with fullgraph and a counter, runs it on
            # Parameters with 1, 10, 11, 15, 15 rows and returns the counter.
            cnt = torch._dynamo.testing.CompileCounter()

            opt = torch.compile(foo, backend=cnt, fullgraph=True)

            x = torch.nn.Parameter(torch.randn(1, 3))
            opt(x)
            x = torch.nn.Parameter(torch.randn(10, 3))
            opt(x)
            x = torch.nn.Parameter(torch.randn(11, 3))
            opt(x)
            x = torch.nn.Parameter(torch.randn(15, 3))
            opt(x)
            x = torch.nn.Parameter(torch.randn(15, 3))
            opt(x)

            return cnt

        @patch.object(torch._dynamo.config, "force_parameter_static_shapes", True)
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_static_comp_default_param():
            # [FUNCTION DOC] HELPER: Runs the scenario with static parameter shapes forced and
            # no automatic dynamic shapes.
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "force_parameter_static_shapes", True)
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_dynamic_comp_default_param():
            # [FUNCTION DOC] HELPER: Runs the scenario with static parameter shapes forced and
            # automatic dynamic shapes on.
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "force_parameter_static_shapes", False)
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_static_comp_dynamic_param():
            # [FUNCTION DOC] HELPER: Runs the scenario with parameter shapes not forced static
            # and no automatic dynamic shapes.
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "force_parameter_static_shapes", False)
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_dynamic_comp_dynamic_param():
            # [FUNCTION DOC] HELPER: Runs the scenario with parameter shapes not forced static
            # and automatic dynamic shapes on.
            return run_foo_6_times_and_count_recompiles()

        torch._dynamo.reset()
        static_comp_default_param = run_static_comp_default_param()
        self.assertEqual(static_comp_default_param.frame_count, 4)
        self.assertEqual(static_comp_default_param.op_count, 4)

        torch._dynamo.reset()
        dynamic_comp_default_param = run_dynamic_comp_default_param()
        self.assertEqual(dynamic_comp_default_param.frame_count, 4)
        self.assertEqual(dynamic_comp_default_param.op_count, 4)

        torch._dynamo.reset()
        static_comp_dynamic_param = run_static_comp_dynamic_param()
        self.assertEqual(static_comp_dynamic_param.frame_count, 4)
        self.assertEqual(static_comp_dynamic_param.op_count, 4)

        torch._dynamo.reset()
        dynamic_comp_dynamic_param = run_dynamic_comp_dynamic_param()
        self.assertEqual(dynamic_comp_dynamic_param.frame_count, 2)
        self.assertEqual(dynamic_comp_dynamic_param.op_count, 2)

    def test_simple_module_recompile(self):
        # [FUNCTION DOC] A module with dropout and linear layers is compiled with fullgraph;
        # alternating eval and train mode twenty times must lead to just 2 compiled frames (one
        # per mode).
        class SimpleDropout(torch.nn.Module):
            # [CLASS DOC] HELPER: Small module with a dropout layer of rate 0.5 and a linear
            # layer from 10 to 1 features.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds the dropout and linear submodules.
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)
                self.linear = torch.nn.Linear(10, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer then dropout.
                return self.dropout(self.linear(x))

        model = SimpleDropout()
        x = torch.randn(10)
        counter = torch._dynamo.testing.CompileCounter()
        model = torch.compile(model, backend=counter, fullgraph=True)
        for _ in range(20):
            model.eval()
            model(x)
            model.train()
            model(x)
        self.assertEqual(counter.frame_count, 2)

    @patch.object(torch._dynamo.config, "recompile_limit", 2)
    def test_no_recursive_compile_after_cache_limit_hit(self):
        # [FUNCTION DOC] With recompile_limit 2 and static shapes, calling a function chain f,
        # g, h with an int argument that changes ten times must give only 2 compiled frames;
        # after the limit is hit the nested calls are not compiled separately.
        def f(x, n):
            # [FUNCTION DOC] HELPER: Adds n then calls g.
            x = x + n
            return g(x, n)

        def g(x, n):
            # [FUNCTION DOC] HELPER: Adds n then calls h.
            x = x + n
            return h(x, n)

        def h(x, n):
            # [FUNCTION DOC] HELPER: Returns x plus n.
            return x + n

        counter = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=counter, dynamic=False)
        for i in range(10):
            opt_f(torch.ones(3), i)
        self.assertEqual(counter.frame_count, 2)

    def test_automatic_dynamic_on_closed_ints(self):
        # [FUNCTION DOC] Closures that capture different Python ints are passed into a compiled
        # function ten times; automatic dynamic handling of the closed-over int yields only 2
        # frames.
        def f(x):
            # [FUNCTION DOC] HELPER: Factory: returns an inner function that adds the captured
            # tensor argument to its own.
            def g(y):
                # [FUNCTION DOC] HELPER: Inner closure: adds the captured value to y.
                return y + x

            return g

        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter)
        def h(x, g):
            # [FUNCTION DOC] HELPER: Compiled function: calls the passed function with the
            # tensor.
            return g(x)

        for i in range(10):
            h(torch.randn(5), f(i))
        self.assertEqual(counter.frame_count, 2)

    @patch.object(torch._dynamo.config, "recompile_limit", 2)
    def test_run_mode_after_cache_limit_hit(self):
        # [FUNCTION DOC] With recompile_limit 2, checks that after the cache limit is hit the
        # function runs eagerly (compile-time-only additions skipped) while earlier compiled
        # shapes still run compiled. Verifies values for n 0 to 3 and again for n 0 and 1, with
        # 2 frames total.
        def f(x, n):
            # [FUNCTION DOC] HELPER: Adds n, adds 1 only while compiling, then calls g.
            x = x + n
            if torch._dynamo.is_compiling():
                x = x + 1
            return g(x, n)

        def g(x, n):
            # [FUNCTION DOC] HELPER: Adds n, adds 2 only while compiling, returns.
            x = x + n
            if torch._dynamo.is_compiling():
                x = x + 2
            return x

        counter = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=counter, dynamic=False)
        # compiles
        self.assertEqual(opt_f(torch.ones(3), 0), torch.ones(3) + 3)
        self.assertEqual(opt_f(torch.ones(3), 1), torch.ones(3) + 5)
        # cache limit hit
        self.assertEqual(opt_f(torch.ones(3), 2), torch.ones(3) + 4)
        self.assertEqual(opt_f(torch.ones(3), 3), torch.ones(3) + 6)
        # run mode
        self.assertEqual(opt_f(torch.ones(3), 0), torch.ones(3) + 3)
        self.assertEqual(opt_f(torch.ones(3), 1), torch.ones(3) + 5)
        self.assertEqual(counter.frame_count, 2)

    @torch._dynamo.config.patch(automatic_dynamic_shapes_mark_as="unbacked")
    def test_automatic_dynamic_shapes_mark_as_unbacked(self):
        # [FUNCTION DOC] With automatic dynamic shapes marking as unbacked, calling with sizes
        # 3, 2, 1, 0 compiles only 2 frames (not three or four).
        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function: squares the tensor.
            return x * x

        f(torch.randn(3))
        f(torch.randn(2))
        f(torch.randn(1))
        f(torch.randn(0))

        self.assertEqual(counter.frame_count, 2)  # not three or four!

    @torch._dynamo.config.patch(recompile_limit=2, fail_on_recompile_limit_hit=True)
    def test_tensorify_python_builtin_mul_does_not_recompile(self):
        # [FUNCTION DOC] With recompile_limit 2 and fail-on-limit, a fullgraph function that
        # multiplies by a scalar derived from a changing float learning rate must not recompile
        # per value: eight calls compare compiled and eager outputs and the backend frame count
        # is at most 2.
        counter = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        def scaling_step(update, dummy_tensor, lr):
            # [FUNCTION DOC] HELPER: Function under test: scales a tensor in place by half the
            # rate, then scales the update by the square root of max(1, rows/cols) times rate;
            # returns the update.
            dummy_tensor.mul_(0.5 * lr)

            r, c = update.size(-2), update.size(-1)
            scaling_factor = max(1, r / c) ** 0.5
            update.mul_(scaling_factor * lr)

            return update

        compiled = torch.compile(scaling_step, backend=counter, fullgraph=True)
        base_update = torch.randn(128, 128)
        base_dummy = torch.randn(324, 64)

        for i in range(8):
            lr = 1e-4 * (i + 1)
            self.assertEqual(
                compiled(base_update.clone(), base_dummy.clone(), lr),
                scaling_step(base_update.clone(), base_dummy.clone(), lr),
            )

        self.assertLessEqual(counter.frame_count, 2)

    @torch._dynamo.config.patch(recompile_limit=2, fail_on_recompile_limit_hit=True)
    def test_tensorify_python_builtin_pow_does_not_recompile(self):
        # [FUNCTION DOC] With recompile_limit 2 and fail-on-limit, a fullgraph function using a
        # changing float raised to a power must not recompile for each value over eight steps;
        # outputs match eager and frames are at most 2.
        counter = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        def step_param(v_t, bc2):
            # [FUNCTION DOC] HELPER: Function under test: square root of the tensor, divided in
            # place by the square root of the scalar, plus a small epsilon.
            eps = 1e-8
            return v_t.sqrt().div_(bc2**0.5).add_(eps)

        compiled = torch.compile(step_param, backend=counter, fullgraph=True)
        base_v_t = torch.randn(64, 1280)

        for step in range(1, 9):
            bc2 = 1.0 - 0.999**step
            self.assertEqual(
                compiled(base_v_t.clone(), bc2),
                step_param(base_v_t.clone(), bc2),
            )

        self.assertLessEqual(counter.frame_count, 2)

    def test_ambient_autocast_recompile(self):
        # [FUNCTION DOC] A fullgraph matmul with a global weight is called outside autocast,
        # under CPU float16 autocast and under CPU bfloat16 autocast; results have the matching
        # dtypes and 3 frames are compiled (autocast state is guarded).
        weights = torch.randn(10, 10)
        counter = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function: matrix product of x with the global
            # weights.
            return torch.mm(x, weights)

        x = torch.randn(1, 10)

        self.assertEqual(fn(x).dtype, torch.float32)

        with torch.autocast("cpu", torch.float16):
            self.assertEqual(fn(x).dtype, torch.float16)

        with torch.autocast("cpu", torch.bfloat16):
            self.assertEqual(fn(x).dtype, torch.bfloat16)

        # should recompile each time
        self.assertEqual(counter.frame_count, 3)

    def test_autocast_constant_fold(self):
        # [FUNCTION DOC] Checks that a function branching on the CPU autocast dtype gives the
        # same result as eager under float16 and bfloat16 autocast and compiles 2 frames,
        # showing the autocast state is guarded.
        # test that constant-folded autocast functions
        # work properly - it should work if the global autocast
        # state is guarded.

        weights = torch.randn(10, 10)
        counter = torch._dynamo.testing.CompileCounterWithBackend("eager")

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test: adds 1 if the CPU autocast dtype is
            # float16, else subtracts 1, then multiplies by the global weights.
            if torch.get_autocast_dtype("cpu") == torch.float16:
                x = x + 1
            else:
                x = x - 1
            return torch.mm(x, weights)

        opt_fn = torch.compile(fn, backend=counter, fullgraph=True)

        x = torch.randn(1, 10)

        with torch.autocast("cpu", torch.float16):
            self.assertEqual(fn(x), opt_fn(x))

        with torch.autocast("cpu", torch.bfloat16):
            self.assertEqual(fn(x), opt_fn(x))

        self.assertEqual(counter.frame_count, 2)

    def test_dunder_call_recompile(self):
        # [FUNCTION DOC] Two instances of a callable class share one compile, but after the
        # class call method is replaced (patched to add 2), calling an existing instance
        # recompiles (frame count goes 1 to 2).
        class Foo:
            # [CLASS DOC] HELPER: Callable class whose call adds 1 to its input; the test later
            # replaces the call method on the class.
            def __call__(self, x):
                # [FUNCTION DOC] HELPER: Adds 1 to x.
                return x + 1

        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter)
        def f(x, foo):
            # [FUNCTION DOC] HELPER: Compiled function: calls the given callable on x.
            return foo(x)

        x = torch.ones(2)
        foo1 = Foo()
        foo2 = Foo()

        # no recompilation
        f(x, foo1)
        f(x, foo2)
        self.assertEqual(counter.frame_count, 1)

        # one recompilation
        Foo.__call__ = lambda self, x: x + 2
        f(x, foo1)
        self.assertEqual(counter.frame_count, 2)

    def test_no_recompile_over_unused_objects(self):
        # [FUNCTION DOC] Regression test imitating a third-party patch application: a compiled
        # function loops over key and patch pairs, collecting patches, then calls f. Different
        # unused key and patch values do not cause recompiles (1 frame).
        # This is a regression test case that imitates
        # https://github.com/city96/ComfyUI-GGUF/blob/47bec6147569a138dd30ad3e14f190a36a3be456/ops.py#L169-L182
        counter = torch._dynamo.testing.CompileCounter()

        def f(x, key, patches):
            # [FUNCTION DOC] HELPER: Callee ignoring key and patches, returns x squared plus 1.
            return x * x + 1

        @torch.compile(backend=counter, fullgraph=True)
        def apply_patches(f, x, keys):
            # [FUNCTION DOC] HELPER: Compiled function: collects patches from the key list,
            # calls f with the last key and the collected patches, and returns the result.
            patches = []
            for key, patch in keys:  # noqa: F402
                patches.append(patch)
            x = f(x, key, patches)
            return x

        # no recompilation
        x = torch.rand(10)
        apply_patches(f, x, [("a", 1), ("b", 2)])
        self.assertEqual(counter.frame_count, 1)
        apply_patches(f, x, [("c", 3), ("d", 4)])
        self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
