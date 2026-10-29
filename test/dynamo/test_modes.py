# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for how torch.compile interacts with torch function modes, torch
# dispatch modes, default-device contexts, and the invoke_subgraph backend used for nested
# compilation under make_fx or AOT export. Defines the module level accelerator device type string
# (current accelerator type, else cpu), two small torch function modes, and four test classes.
# Importing has no side effects beyond that.

import operator
import unittest
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._C import (
    _len_torch_function_stack,
    _pop_torch_function_stack,
    _push_on_torch_function_stack,
)
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch.overrides import (
    _get_current_function_mode_stack,
    BaseTorchFunctionMode,
    TorchFunctionMode,
)
from torch.testing._internal.common_device_type import (
    IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED,
)
from torch.testing._internal.common_utils import skipIfXpu, TEST_ACCELERATOR
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils._device import DeviceContext
from torch.utils._python_dispatch import TorchDispatchMode


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class TestMode(BaseTorchFunctionMode):
    # [CLASS DOC] Torch function mode that replaces every add call with a fresh two by two zeros
    # tensor and forwards all other functions to the base mode. Stateless.
    def __torch_function__(self, func, types, args, kwargs=None):
        # [FUNCTION DOC] Mode hook. Normalizes empty kwargs to a mapping; returns a new two by two
        # zeros tensor when the intercepted function is the torch add function, else defers to the
        # base class implementation with the same arguments.
        if not kwargs:
            kwargs = {}

        if func == torch.add:
            return torch.zeros(2, 2)

        return super().__torch_function__(func, types, args, kwargs)


class HopDetectionError(Exception):
    # [CLASS DOC] HELPER: Exception type raised by TestModeRaises to prove that a higher order
    # operation was seen by the mode.
    pass


class TestModeRaises(BaseTorchFunctionMode):
    # [CLASS DOC] Torch function mode that raises HopDetectionError when it intercepts the
    # flex_attention higher order operation, and otherwise forwards to the base mode.
    def __torch_function__(self, func, types, args, kwargs=None):
        # [FUNCTION DOC] Mode hook. Normalizes empty kwargs, imports the higher order ops package,
        # raises HopDetectionError with message test if the function is the flex_attention higher
        # order op, else defers to the base class.
        if not kwargs:
            kwargs = {}

        import torch._higher_order_ops

        if func == torch._higher_order_ops.flex_attention:
            raise HopDetectionError("test")

        return super().__torch_function__(func, types, args, kwargs)


class TorchDispatchModeTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for torch dispatch modes active while compiling: compile-internal
    # ignoring, skipping compilation under user modes, and graph capture of dispatch mode
    # stack queries.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup hook; only calls the parent implementation.
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown hook; only calls the parent implementation.
        super().tearDownClass()

    def test_torch_dispatch_ignore_compile_internals(self):
        # [FUNCTION DOC] Registers a custom op and a dispatch mode that opts out of seeing
        # compile internals and records an absolute-sum checksum whenever the custom op runs.
        # Under that mode, runs the op, an inductor-compiled function that must not graph break,
        # and the op again. Expects exactly two checksums (the compiled function must not
        # trigger the custom op) and one total and one ok compiled frame in the Dynamo counters.
        # Clears the counters first.
        counters.clear()
        from torch.utils._python_dispatch import TorchDispatchMode

        @torch.library.custom_op("mylib::modes_checksum", mutates_args=())
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op body returning a clone of its tensor input.
            return x.clone()

        def checksum(x):
            # [FUNCTION DOC] HELPER: Returns the sum of absolute values of a tensor.
            return x.abs().sum()

        _checksums = []

        class ChecksumFoo(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that declares compile internals ignorable and
            # records a checksum of the first argument to the modes_checksum custom op into the
            # enclosing list. Always forwards the call.
            @classmethod
            def ignore_compile_internals(cls):
                # [FUNCTION DOC] HELPER: Class level hook returning true to tell the framework
                # this mode should be ignored for compile-internal operations.
                return True

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the parent constructor.
                super().__init__()

            def __torch_dispatch__(self, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook. Normalizes missing kwargs, appends a
                # checksum of the first positional argument when the intercepted op is the default
                # overload of the custom op, then runs the op unchanged and returns its result.
                kwargs = kwargs or {}

                if func is torch.ops.mylib.modes_checksum.default:
                    # Do some compute, smoketest to see if there's a bad interaction
                    _checksums.append(args[0].abs().sum())

                return func(*args, **kwargs)

        # test e2e, with Inductor, as smoketest.
        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend="inductor")
        def g(x):
            # [FUNCTION DOC] HELPER: Returns twice the cosine of the sine of the input.
            return 2 * x.sin().cos()

        x = torch.randn(3)

        with ChecksumFoo():
            foo(x)
            g(x)
            foo(x)

        self.assertEqual(len(_checksums), 2)
        # The correct result here is 1: Dynamo should capture the `g` frame.
        self.assertEqual(counters["frames"]["total"], 1)
        self.assertEqual(counters["frames"]["ok"], 1)

    def test_skip_torch_dispatch_modes(self):
        # [FUNCTION DOC] Checks that compilation is skipped (no frames captured) while a user
        # dispatch mode that rewrites add into mul is active, and that eager and compiled results
        # agree under the mode. After the mode exits, the compiled function runs normally, matches
        # eager, and captures one frame.
        class RewriteAddToMul(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that swaps the ATen add overload for the
            # mul overload.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook; replaces the add tensor op with the mul
                # tensor op and runs the op with the given arguments, returning its result.
                if func is torch.ops.aten.add.Tensor:
                    func = torch.ops.aten.mul.Tensor
                return func(*args, **kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input added to itself.
            return x + x

        cnt = torch._dynamo.testing.CompileCounter()

        x = torch.tensor([3.0])
        compiled_fn = torch.compile(fn, backend=cnt)
        with RewriteAddToMul():
            eager_res = fn(x)
            compiled_res = compiled_fn(x)

        self.assertEqual(eager_res, compiled_res)
        self.assertEqual(cnt.frame_count, 0)

        self.assertEqual(compiled_fn(x), fn(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_flop_counter_mode_compile_skip_is_transient(self):
        # [FUNCTION DOC] Checks that the compile skip caused by an active flop counter mode lasts
        # only while the mode is active: under the mode the compiled function matches eager with
        # zero frames; afterwards it matches eager and captures one frame.
        from torch.utils.flop_counter import FlopCounterMode

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the sine of a matrix product of two inputs.
            return (x @ y).sin()

        cnt = torch._dynamo.testing.CompileCounter()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with FlopCounterMode(display=False):
            compiled_fn = torch.compile(fn, backend=cnt)
            self.assertEqual(compiled_fn(x, y), fn(x, y))

        self.assertEqual(cnt.frame_count, 0)
        self.assertEqual(compiled_fn(x, y), fn(x, y))
        self.assertEqual(cnt.frame_count, 1)

    def test_get_current_dispatch_mode_stack_no_graph_break(self):
        # [FUNCTION DOC] Checks that querying the current dispatch mode stack inside a full-graph
        # compiled function causes no graph break, that the stack is empty at trace time so the
        # output equals the ones tensor, and that one frame was captured.
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds the length of the current dispatch mode stack to
            # the input.
            stack = _get_current_dispatch_mode_stack()
            return x + len(stack)

        x = torch.ones(2, 2)
        result = fn(x)
        # Stack is empty during tracing, so len([]) == 0 and result == ones(2,2)
        self.assertEqual(result, torch.ones(2, 2))
        # fullgraph=True would have raised if a graph break occurred
        self.assertEqual(cnt.frame_count, 1)


class TorchFunctionModeTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for torch function mode stack handling in Dynamo: guards on mode stacks,
    # stack push, pop and length operations traced inside compiled code, default device
    # interaction, graph breaks, exceptions, and mode-intercepted builtin operators. Class state:
    # the previous default device and whether a global default device context existed are saved at
    # class setup and restored at teardown.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup hook. Saves the current default device (cpu if the getter is
        # unavailable) and whether a global default device context was present, then calls the
        # parent setup.
        try:
            cls.default_device_old = torch.get_default_device()
        except AttributeError:
            cls.default_device_old = torch.device("cpu")
        global_default_ctx = getattr(
            getattr(torch, "_GLOBAL_DEVICE_CONTEXT", None), "device_context", None
        )
        cls._had_global_default_device = global_default_ctx is not None
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown hook. Restores the saved default device only if a global
        # default device context existed at class setup, then calls the parent teardown.
        if cls._had_global_default_device:
            torch.set_default_device(cls.default_device_old)
        super().tearDownClass()

    def setUp(self):
        # [FUNCTION DOC] Per-test setup. Calls the parent setup, clears the default device and
        # resets Dynamo state.
        super().setUp()
        torch.set_default_device(None)
        torch._dynamo.reset()

    def tearDown(self):
        # [FUNCTION DOC] Per-test teardown. Calls the parent teardown, clears the default device
        # and resets Dynamo state.
        super().tearDown()
        torch.set_default_device(None)
        torch._dynamo.reset()

    def _run_torch_function_mode_guard_test(self):
        # [FUNCTION DOC] HELPER: Shared body for the guard tests. Compiles an add-one function
        # with a frame counting backend and runs it with no mode (one frame), with the first mode
        # (recompile, two), with both modes in one nesting order (three), in the reverse order
        # (four), and again with only the first mode (cache hit, still four). Verifies stack
        # contents are guarded on.
        class TestMode1(BaseTorchFunctionMode):
            # [CLASS DOC] HELPER: Empty base torch function mode subclass used to
            # distinguish stacks.
            pass

        class TestMode2(BaseTorchFunctionMode):
            # [CLASS DOC] HELPER: Second empty base torch function mode subclass.
            pass

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt.__call__)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one.
            return x + 1

        inp = torch.ones(2, 2)
        fn(inp)
        self.assertEqual(cnt.frame_count, 1)

        with TestMode1():
            fn(inp)
        self.assertEqual(cnt.frame_count, 2)

        with TestMode1(), TestMode2():
            fn(inp)
        self.assertEqual(cnt.frame_count, 3)

        with TestMode2(), TestMode1():
            fn(inp)
        self.assertEqual(cnt.frame_count, 4)

        with TestMode1():
            fn(inp)
        self.assertEqual(cnt.frame_count, 4)

    @torch._dynamo.config.patch("enable_cpp_guard_manager", False)
    def test_torch_function_mode_guards_py(self):
        # [FUNCTION DOC] Runs the shared mode guard test with the C++ guard manager disabled by a
        # config patch.
        self._run_torch_function_mode_guard_test()

    def test_torch_function_mode_guards_cpp(self):
        # [FUNCTION DOC] Runs the shared mode guard test with the default (C++) guard manager.
        self._run_torch_function_mode_guard_test()

    def test_torch_function_mode_preserves_cuda_rng_state(self):
        # [FUNCTION DOC] Checks that a function compiled with the eager optimize helper can enter
        # a torch function mode that returns a constant for every intercepted function and still
        # return the plain Python value from the with block.
        # KNOWN ISSUE: the name mentions CUDA RNG state, but the test never touches RNG state.
        class ConstantReturnMode(TorchFunctionMode):
            # [CLASS DOC] HELPER: Torch function mode returning the constant minus forty-two for
            # any function.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook ignoring its arguments and returning
                # minus forty-two.
                return -42

        @torch._dynamo.optimize("eager")
        def fn():
            # [FUNCTION DOC] HELPER: Enters the constant mode and returns the integer 123
            # from within.
            with ConstantReturnMode():
                return 123

        self.assertEqual(fn(), 123)

    def test_stack_state_mutation_default_device(self):
        # [FUNCTION DOC] Checks default-device handling across a compiled stack mutation: with
        # two base modes active, a full-graph function sets the default device to cpu and pops
        # the top mode. After pushing the popped mode back, the mode stack must be a cpu device
        # context, then the first mode, then the second. Always resets the default device to
        # none at the end.
        m = BaseTorchFunctionMode()
        m1 = BaseTorchFunctionMode()
        try:
            with m, m1:

                @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
                def fn(x):
                    # [FUNCTION DOC] HELPER: Sets the default device to cpu and pops the torch
                    # function stack; returns nothing.
                    torch.set_default_device("cpu")
                    _pop_torch_function_stack()

                fn(torch.ones(2, 2))
                _push_on_torch_function_stack(m1)

                stack = _get_current_function_mode_stack()
                self.assertIsInstance(stack[0], DeviceContext)
                self.assertEqual(stack[0].device, torch.device("cpu"))
                self.assertIs(stack[1], m)
                self.assertIs(stack[2], m1)
        finally:
            torch.set_default_device(None)

    def test_stack_state_clear_default_device(self):
        # [FUNCTION DOC] Checks how setting and clearing the default device inside full-graph
        # compiled functions changes the mode stack. First clears the device with an empty stack
        # (stack stays empty). Then with two modes active, sets cpu, clears, sets cpu (device
        # context ends up at the stack bottom). Then with a global device set, clears it inside
        # compiled code (stack is just the two modes). Finally sets cpu twice and checks a single
        # cpu device context, and resets the device.
        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def fn(x):
            # [FUNCTION DOC] HELPER: Clears the default device and returns the input plus one.
            torch.set_default_device(None)
            return x + 1

        fn(torch.ones(2, 2))
        stack = _get_current_function_mode_stack()
        self.assertEqual(len(stack), 0)

        m = BaseTorchFunctionMode()
        m1 = BaseTorchFunctionMode()

        # Stack populated, add device
        with m, m1:

            @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
            def fn(x):
                # [FUNCTION DOC] HELPER: Sets cpu, clears, sets cpu as default device; returns
                # input plus one.
                torch.set_default_device("cpu")
                torch.set_default_device(None)
                torch.set_default_device("cpu")
                return x + 1

            fn(torch.ones(2, 2))
            stack = _get_current_function_mode_stack()
            self.assertEqual(stack[0].device, torch.device("cpu"))
            self.assertIs(stack[1], m)
            self.assertIs(stack[2], m1)

        # Stack populated, remove device
        torch.set_default_device("cpu")
        with m, m1:

            @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
            def fn(x):
                # [FUNCTION DOC] HELPER: Clears the default device; returns input plus one.
                torch.set_default_device(None)
                return x + 1

            fn(torch.ones(2, 2))
            stack = _get_current_function_mode_stack()
            self.assertIs(stack[0], m)
            self.assertIs(stack[1], m1)

        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def fn(x):
            # [FUNCTION DOC] HELPER: Sets the cpu default device twice; returns input plus one.
            torch.set_default_device("cpu")
            torch.set_default_device("cpu")
            return x + 1

        fn(torch.ones(2, 2))
        stack = _get_current_function_mode_stack()
        self.assertEqual(stack[0].device, torch.device("cpu"))
        torch.set_default_device(None)

    def test_pop_torch_function_mode(self):
        # [FUNCTION DOC] Checks a full-graph compiled function can pop the only active mode off
        # the torch function stack: after the call the stack length is zero. The test pushes the
        # mode back before leaving the with block so exit does not fail, and checks the length is
        # zero again after exit.
        m = BaseTorchFunctionMode()
        with m:

            @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
            def fn(x):
                # [FUNCTION DOC] HELPER: Pops the torch function stack and returns the input
                # plus one.
                _pop_torch_function_stack()
                return x + 1

            fn(torch.ones(2, 2))

            self.assertEqual(_len_torch_function_stack(), 0)
            # reset stack so __exit__ doesn't crash
            _push_on_torch_function_stack(m)

        self.assertEqual(_len_torch_function_stack(), 0)

    def test_is_torch_function_all_disabled(self):
        # [FUNCTION DOC] Checks that the flag reporting all torch function handling as disabled is
        # false when read inside a full-graph compiled function.
        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a pair of the all-disabled flag and an add of the
            # input and one point zero.
            return (
                torch._C._is_torch_function_all_disabled(),
                torch.add(x, 1.0),
            )

        input = torch.ones(2, 2)
        res, _ = fn(input)
        self.assertFalse(res)

    def test_error_empty_stack_pop_torch_function_mode(self):
        # [FUNCTION DOC] Checks that popping the torch function stack when it is empty inside a
        # full-graph compile raises Dynamo's Unsupported error with a message about popping from
        # an empty torch function mode stack.
        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def fn(x):
            # [FUNCTION DOC] HELPER: Pops the (empty) torch function stack and returns input
            # plus one.
            _pop_torch_function_stack()
            return x + 1

        self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Attempted to pop from empty torch function mode stack",
            lambda: fn(torch.ones(2, 2)),
        )

    def test_push_torch_function_mode(self):
        # [FUNCTION DOC] Checks that pushing a mode object onto the torch function stack inside a
        # full-graph compiled function leaves the stack length at two when one mode was already
        # active. Pops the extra mode to restore state and checks length zero after exit.
        m = BaseTorchFunctionMode()
        with m:

            @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
            def fn(x, m):
                # [FUNCTION DOC] HELPER: Pushes the given mode onto the torch function stack and
                # returns the input plus one.
                _push_on_torch_function_stack(m)
                return x + 1

            fn(torch.ones(2, 2), m)

            self.assertEqual(_len_torch_function_stack(), 2)
            # reset stack state
            _pop_torch_function_stack()

        self.assertEqual(_len_torch_function_stack(), 0)

    def test_len_torch_function_mode(self):
        # [FUNCTION DOC] Checks that the stack length query inside a full-graph compile reads the
        # real stack: with one mode active, the result is the ones matrix plus one, and the stack
        # length after the call is still one.
        m = BaseTorchFunctionMode()
        with m:

            @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
            def fn(x):
                # [FUNCTION DOC] HELPER: Adds the current torch function stack length to
                # the input.
                z = _len_torch_function_stack()
                return x + z

            res = fn(torch.ones(2, 2))
            self.assertEqual(res, torch.ones(2, 2) + 1)
            self.assertEqual(_len_torch_function_stack(), 1)

    def test_intermedate_torch_function_mode_construction_mutation(self):
        # [FUNCTION DOC] Checks that constructing a torch function mode object inside a
        # full-graph compiled function and mutating its attributes, then returning it, compiles
        # without error.
        class TestMode(BaseTorchFunctionMode):
            # [CLASS DOC] HELPER: Torch function mode storing one constructor argument as
            # an attribute.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores the argument as attribute x without calling the
                # parent constructor.
                self.x = x

        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates the mode with argument two, sets another attribute y
            # on it and returns the input plus one together with the mode.
            z = TestMode(2)
            z.y = 2
            return x + 1, z

        fn(torch.ones(2, 2))

    def test_torch_function_mode_enabled_guard(self):
        # [FUNCTION DOC] Checks that the guard on whether torch function handling is enabled
        # causes a recompile: inside an active mode and subclass-disable context, the function
        # runs once with all handling disabled and once with only subclass handling disabled; two
        # frames expected.
        cnt = torch._dynamo.testing.CompileCounter()
        inp = torch.ones(2, 2)

        @torch.compile(backend=cnt.__call__)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one.
            return x + 1

        with BaseTorchFunctionMode(), torch._C.DisableTorchFunctionSubclass():
            with torch._C.DisableTorchFunction():
                fn(inp)
            fn(inp)
        self.assertEqual(cnt.frame_count, 2)

    def test_nested_torch_function_mode(self):
        # [FUNCTION DOC] Checks compiled and eager behavior match with two nested custom modes.
        # Mode one returns zeros for add and mode two returns ones for mul, each flagging when it
        # was called. For an add-only function and a mul-plus-add function, compares result and
        # both flags between eager and full-graph compile, resetting the flags between runs.
        mode_1_called = False
        mode_2_called = False

        def reset_state():
            # [FUNCTION DOC] HELPER: Clears the two enclosing flags marking whether each mode
            # was called.
            nonlocal mode_1_called
            nonlocal mode_2_called
            mode_1_called = False
            mode_2_called = False

        ones = torch.ones(2, 2)
        zeros = torch.zeros(2, 2)

        class TestMode1(BaseTorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that marks the enclosing flag for mode one and returns the
            # enclosing zeros tensor for add.
            def __torch_function__(self, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook; normalizes kwargs, sets the mode one called
                # flag, returns the zeros tensor for add, otherwise defers to base.
                if not kwargs:
                    kwargs = {}

                nonlocal mode_1_called

                mode_1_called = True

                if func == torch.add:
                    return zeros

                return super().__torch_function__(func, types, args, kwargs)

        class TestMode2(BaseTorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that marks the enclosing flag for mode two and returns the
            # enclosing ones tensor for mul.
            def __torch_function__(self, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook; normalizes kwargs, sets the mode two called
                # flag, returns the ones tensor for mul, otherwise defers to base.
                if not kwargs:
                    kwargs = {}

                nonlocal mode_2_called

                mode_2_called = True

                if func == torch.mul:
                    return ones

                return super().__torch_function__(func, types, args, kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns add of the input and three.
            return torch.add(x, 3)

        def fn_2(x):
            # [FUNCTION DOC] HELPER: Returns mul of the input and three plus add of the input
            # and three.
            return torch.mul(x, 3) + torch.add(x, 3)

        inp = torch.ones(2, 2) + 1

        for fn_i in [fn, fn_2]:
            fn_opt = torch.compile(fn_i, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
            with TestMode1(), TestMode2():
                expected = fn_i(inp), mode_1_called, mode_2_called
                reset_state()
                actual = fn_opt(inp), mode_1_called, mode_2_called
                reset_state()

            self.assertEqual(expected, actual)

    def test_torch_function_mode_disable(self):
        # [FUNCTION DOC] Checks that disabling torch function handling, either for subclasses
        # only or fully, makes compiled output equal eager output while a mode is active and
        # the input is a tensor subclass whose handler replaces add with ones. Compares both
        # disable contexts.
        class TestSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose torch function handler returns a ones
            # matrix for add and defers otherwise.
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Class level handler; normalizes kwargs, returns a ones
                # matrix for add, otherwise defers to the parent.
                if not kwargs:
                    kwargs = {}
                if func == torch.add:
                    return torch.ones(2, 2)
                return super().__torch_function__(func, types, args, kwargs)

        class TestMode(BaseTorchFunctionMode):
            # [CLASS DOC] HELPER: Mode returning a zeros matrix for add and deferring otherwise.
            def __torch_function__(self, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook; normalizes kwargs, returns zeros for add,
                # else defers.
                if not kwargs:
                    kwargs = {}

                if func == torch.add:
                    return torch.zeros(2, 2)

                return super().__torch_function__(func, types, args, kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns add of the input and three.
            return torch.add(x, 3)

        inp = (torch.ones(2, 2) + 1).as_subclass(TestSubclass)

        fn_opt = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        with TestMode():
            with torch._C.DisableTorchFunctionSubclass():
                expected = fn(inp)
                actual = fn_opt(inp)

            self.assertEqual(expected, actual)

            with torch._C.DisableTorchFunction():
                expected = fn(inp)
                actual = fn_opt(inp)

            self.assertEqual(expected, actual)

    def test_torch_function_mode_highest_priority(self):
        # [FUNCTION DOC] Checks that when a mode and a subclass both handle add, the compiled
        # result equals the eager result (modes take priority over the subclass handler).
        class TestSubclass(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose torch function handler returns ones
            # for add.
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Class level handler; normalizes kwargs, returns ones for
                # add, else defers to the parent.
                if not kwargs:
                    kwargs = {}
                if func == torch.add:
                    return torch.ones(2, 2)
                return super().__torch_function__(func, types, args, kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns add of the input and three.
            return torch.add(x, 3)

        inp = (torch.ones(2, 2) + 1).as_subclass(TestSubclass)

        fn_opt = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        with TestMode():
            expected = fn(inp)
            actual = fn_opt(inp)

        self.assertEqual(expected, actual)

    def test_torch_function_mode_enter_exit(self):
        # [FUNCTION DOC] Checks that entering and leaving a mode inside a full-graph compiled
        # function works: the add inside the mode is intercepted, the add outside is not, and the
        # result equals eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds three under the module level mode, then adds the second
            # input outside it.
            with TestMode():
                o = torch.add(x, 3)

            return torch.add(o, y)

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2) + 2)
        fn_opt = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        expected = fn(*inp)
        actual = fn_opt(*inp)

        self.assertEqual(expected, actual)

    def test_torch_function_mode_graph_break(self):
        # [FUNCTION DOC] Same as the enter-exit test, but with an explicit graph break inside the
        # mode and without full-graph; the compiled result must equal eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Under the module level mode, graph breaks, adds three, then
            # adds the second input after the mode exits.
            with TestMode():
                torch._dynamo.graph_break()
                o = torch.add(x, 3)

            return torch.add(o, y)

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2) + 2)
        fn_opt = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND

        expected = fn(*inp)
        actual = fn_opt(*inp)

        self.assertEqual(expected, actual)

    def test_torch_function_mode_and_pop_graph_break(self):
        # [FUNCTION DOC] Checks a graph break between popping the mode off the stack and pushing
        # it back keeps compiled output equal to eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Under the module level mode, pops the stack, graph
            # breaks, pushes the popped mode back, adds three, then adds the second input
            # outside the mode.
            with TestMode():
                z = _pop_torch_function_stack()
                torch._dynamo.graph_break()
                _push_on_torch_function_stack(z)
                o = torch.add(x, 3)

            return torch.add(o, y)

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2) + 2)
        fn_opt = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND

        expected = fn(*inp)
        actual = fn_opt(*inp)

        self.assertEqual(expected, actual)

    def test_torch_function_mode_restore_on_exc(self):
        # [FUNCTION DOC] Checks that an exception from a disabled helper inside a mode leaves no
        # mode on the torch function stack: the call is expected to raise a runtime error, which
        # is swallowed, and the stack length must be zero afterwards.
        @torch._dynamo.disable()
        def err():
            # [FUNCTION DOC] HELPER: Disabled for Dynamo; always raises a runtime error with
            # message test.
            raise RuntimeError("test")

        @torch.compile()  # noqa: UNSPECIFIED_BACKEND
        def fn(x):
            # [FUNCTION DOC] HELPER: Under the module level mode, adds one in place, calls the
            # raising helper, then would add two and return.
            with TestMode():
                x += 1
                err()
                x += 2
                return x

        try:
            fn(torch.ones(2, 2))
        except RuntimeError:
            pass
        self.assertEqual(_len_torch_function_stack(), 0)

    def test_torch_function_mode_and_pop_graph_break_mutation(self):
        # [FUNCTION DOC] Like the pop-graph-break test but the popped mode also gets an attribute
        # set before the graph break; that attribute is then used as a multiplier after the mode
        # is pushed back. Compiled output must equal eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Under the module level mode, pops it, sets attribute y to
            # five, graph breaks, pushes it back, adds three, multiplies by the attribute, then
            # adds the second input after the with block.
            with TestMode():
                z = _pop_torch_function_stack()
                z.y = 5
                torch._dynamo.graph_break()
                _push_on_torch_function_stack(z)
                o = torch.add(x, 3)
                o = torch.mul(o, z.y)

            return torch.add(o, y)

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2) + 2)
        fn_opt = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND

        expected = fn(*inp)
        actual = fn_opt(*inp)

        self.assertEqual(expected, actual)

    def test_torch_function_mode_mutated_state(self):
        # [FUNCTION DOC] Checks that mutable state on a mode is guarded: a counting mode
        # increments a counter per intercepted call. After the first compiled call the counter is
        # positive and one frame exists; the second call sees a changed counter and must
        # recompile, giving two frames.
        class CounterMode(TorchFunctionMode):
            # [CLASS DOC] HELPER: Torch function mode with a call counter attribute incremented on
            # each intercepted function, which is then run unchanged.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the counter to zero without calling the
                # parent constructor.
                self.count = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Increments the counter and calls the intercepted function
                # with the given arguments and kwargs, returning its result.
                self.count += 1
                return func(*args, **(kwargs or {}))

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of the input.
            return x.sin()

        with CounterMode() as mode:
            f(torch.randn(3))
            first_count = mode.count
            first_frame_count = cnt.frame_count

            # count changed, should recompile
            f(torch.randn(3))

        self.assertEqual(first_frame_count, 1)
        self.assertGreater(first_count, 0)
        self.assertEqual(cnt.frame_count, 2)

    # Needs larger cache size since we recompile for each op
    @patch.object(torch._dynamo.config, "recompile_limit", 48)
    def test_builtin_equivalent_funcs(self):
        # [FUNCTION DOC] Checks that each operator Dynamo maps to a tensor method equivalent is
        # traced into the expected torch function. For unary, unary integer, binary, binary
        # integer, and tensor-and-integer operator lists, compiles a wrapper and, under a
        # recording mode, verifies the function seen during tracing equals the mapped one. Repeats
        # for reversed operand variants using the reverse map, skipping known problematic
        # operators. Raises the recompile limit by a patch because each operator recompiles.
        from torch._dynamo.variables.builtin import (
            BUILTIN_TO_TENSOR_FN_MAP,
            BUILTIN_TO_TENSOR_RFN_MAP,
        )
        from torch._dynamo.variables.torch_function import (
            bin_int_ops,
            bin_ops,
            tensor_and_int_ops,
            un_int_ops,
            un_ops,
        )

        expected_func = None
        valid = False

        class FuncEquivMode(BaseTorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that, while compiling, records in an enclosing flag
            # whether the intercepted function equals the enclosing expected function, then
            # defers to the base.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook; normalizes kwargs, and if Dynamo is compiling
                # sets the valid flag to whether the function matches the expected one; then
                # defers to the base.
                nonlocal expected_func
                nonlocal valid
                if not kwargs:
                    kwargs = {}
                if torch._dynamo.is_compiling():
                    valid = expected_func == func
                return super().__torch_function__(func, types, args, kwargs)

        inp0 = torch.ones(1, 1)
        inp1 = torch.ones(1, 1)
        inp0_int = torch.ones(1, 1, dtype=torch.int32)
        inp1_int = torch.ones(1, 1, dtype=torch.int32)

        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def fn_un(op, inp):
            # [FUNCTION DOC] HELPER: Applies the given unary operator to the input.
            return op(inp)

        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def fn_un_int(op, inp):
            # [FUNCTION DOC] HELPER: Applies the given unary integer operator to the input.
            return op(inp)

        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def fn_bin(op, inp0, inp1):
            # [FUNCTION DOC] HELPER: Applies the given binary operator to two inputs.
            return op(inp0, inp1)

        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def fn_bin_int(op, inp0, inp1):
            # [FUNCTION DOC] HELPER: Applies the given binary integer operator to two inputs.
            return op(inp0, inp1)

        @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        def fn_tensor_and_int(op, inp0, inp1):
            # [FUNCTION DOC] HELPER: Applies the given operator to a tensor and an integer.
            return op(inp0, inp1)

        setups_and_oplists = [
            (lambda o: fn_un(o, inp0), un_ops),
            (lambda o: fn_un_int(o, inp0_int), un_int_ops),
            (lambda o: fn_bin(o, inp0, inp1), bin_ops),
            (lambda o: fn_bin_int(o, inp0_int, inp1_int), bin_int_ops),
            (lambda o: fn_tensor_and_int(o, inp0_int, 0), tensor_and_int_ops),
        ]

        # gather the reverse functions
        rsetups_and_oplists = [
            (
                lambda o: fn_bin(o, 1, inp1),
                bin_ops,
            ),  # Get r* ops, (ex. __sub__(int, Tensor) -> __rsub__(Tensor, int))
            (lambda o: fn_bin_int(o, 1, inp1_int), bin_int_ops),
            (lambda o: fn_tensor_and_int(o, 0, inp0_int), tensor_and_int_ops),
        ]

        skips = {operator.not_}  # Has local scalar dense call which graph breaks
        rskips = {
            operator.matmul,
            operator.imatmul,
            operator.getitem,
        }  # Doesn't type check with reversed args

        def run_checks(setups_and_oplists, skips, ref_map):
            # [FUNCTION DOC] HELPER: For every operator in each list that is not skipped and has
            # an entry in the reference map, enters the recording mode, sets the expected function
            # from the map, runs the setup callable, asserts the validity flag was set, then
            # resets both flags.
            nonlocal valid
            nonlocal expected_func
            for setup_fn, op_list in setups_and_oplists:
                for op in op_list:
                    if op in skips or op not in ref_map:
                        continue
                    with FuncEquivMode():
                        expected_func = ref_map[op]
                        setup_fn(op)
                        self.assertTrue(valid)

                    expected_func = None
                    valid = False

        run_checks(setups_and_oplists, skips, BUILTIN_TO_TENSOR_FN_MAP)
        run_checks(rsetups_and_oplists, rskips, BUILTIN_TO_TENSOR_RFN_MAP)

    def test_expand(self):
        # [FUNCTION DOC] Regression test for a transformed distribution issue: under a cpu device
        # context, a full-graph compiled function builds a transformed normal distribution with
        # tanh and affine transforms and computes log probability of a reparameterized sample.
        # Passes if compilation and execution succeed.
        from torch.distributions import (
            AffineTransform,
            ComposeTransform,
            Normal,
            TanhTransform,
            TransformedDistribution,
        )

        # https://github.com/pytorch/pytorch/issues/141232
        with torch.device("cpu"):

            @torch.compile(fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
            def func(a):
                # [FUNCTION DOC] HELPER: Builds the transformed distribution from a normal
                # with the input as mean, takes ten reparameterized samples and returns their
                # log probability.
                d = TransformedDistribution(
                    Normal(a, 1),
                    ComposeTransform([TanhTransform(), AffineTransform(2, 2)]),
                )
                b = d.log_prob(d.rsample((10,)))
                return b

            func(torch.randn(3))

    @unittest.skipIf(not HAS_GPU, "requires GPU and Triton")
    def test_flex_attention(self):
        # [FUNCTION DOC] Requires a GPU and Triton. Sets the default device to the accelerator,
        # builds a block mask for a prefix mask function with compile enabled, and always resets
        # the default device afterwards.
        # KNOWN ISSUE: the compiled flex_attention assigned to the local name is never called;
        # only block mask creation is exercised.
        import torch
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        try:
            torch.set_default_device(device_type)

            flex_attention = torch.compile(flex_attention, dynamic=False)  # noqa: UNSPECIFIED_BACKEND

            prefix_lengths = torch.arange(8)

            def prefix_lm(b, h, q, kv):
                # [FUNCTION DOC] HELPER: Mask function returning whether the prefix length of a
                # batch entry is at least the key-value index.
                return prefix_lengths[b] >= kv

            # This runs in fullgraph already
            create_block_mask(
                prefix_lm, 8, None, 512, 512, _compile=True, device=device_type
            )
        finally:
            torch.set_default_device(None)

    def test_register_hook(self):
        # [FUNCTION DOC] Checks that registering a partial-bound gradient hook on an input inside
        # a full-graph compiled module under a cpu device context compiles without error.
        import functools

        def my_hook(grad, *, k=0):
            # [FUNCTION DOC] HELPER: Gradient hook adding the keyword constant k to the gradient.
            return grad + k

        hook = functools.partial(my_hook, k=3)

        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward registers the enclosing hook on the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Registers the hook on the input, multiplies by two then
                # by three, and returns the result in a one-element tuple.
                x.register_hook(hook)
                y = x.mul(2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x = torch.ones(4, requires_grad=True)

        with torch.device("cpu"):
            torch.compile(mod, fullgraph=True)(x)  # noqa: UNSPECIFIED_BACKEND

    def test_tensor_unflatten_with_default_device(self):
        # [FUNCTION DOC] Checks that unflattening the first dimension of a tensor into two by
        # two works under a cpu device context in an eager-backend full-graph compile, and
        # matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Unflattens the first dimension into sizes two and two.
            return x.unflatten(0, (2, 2))

        x = torch.randn(4)
        with torch.device("cpu"):
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

    @unittest.skipIf(not HAS_GPU, "requires GPU and Triton")
    @skipIfXpu(msg="XPU does not support flex attention")
    def test_hop(self):
        # [FUNCTION DOC] Requires GPU and Triton, skipped on XPU. Under the accelerator device,
        # compiles flex_attention and runs it inside the raising mode; expects Dynamo to report
        # an Unsupported error naming the HopDetectionError, proving the mode saw the higher
        # order op.
        import torch
        import torch._higher_order_ops
        from torch.nn.attention.flex_attention import (
            flex_attention as flex_attention_eager,
        )

        with torch.device(device_type):
            flex_attention = torch.compile(flex_attention_eager, dynamic=False)  # noqa: UNSPECIFIED_BACKEND

            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "raised exception HopDetectionError\\('test'\\)",
            ):
                # This runs in fullgraph already
                with TestModeRaises():
                    flex_attention(
                        torch.ones(2, 2, 2, 2),
                        torch.ones(2, 2, 2, 2),
                        torch.ones(2, 2, 2, 2),
                    )

    @unittest.skipIf(not HAS_GPU, "requires GPU and Triton")
    @skipIfXpu(msg="XPU does not support flex attention")
    def test_hop_eager(self):
        # [FUNCTION DOC] Requires GPU and Triton, skipped on XPU. Same as the compiled hop test
        # but runs flex_attention eagerly under the raising mode; expects the same Unsupported
        # error naming HopDetectionError.
        import torch
        import torch._higher_order_ops
        from torch.nn.attention.flex_attention import (
            flex_attention as flex_attention_eager,
        )

        with torch.device(device_type):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "raised exception HopDetectionError\\('test'\\)",
            ):
                with TestModeRaises():
                    flex_attention_eager(
                        torch.ones(2, 2, 2, 2),
                        torch.ones(2, 2, 2, 2),
                        torch.ones(2, 2, 2, 2),
                    )

    @unittest.skipIf(not TEST_ACCELERATOR, "requires accelerator")
    def test_default_device_factory_functions(self):
        # [FUNCTION DOC] Requires an accelerator. Checks factory functions follow the default
        # device in eager-backend full-graph code. With the accelerator as default, random
        # integers and zeros-like with matching device land on it while zeros-like with an
        # explicit cpu device stays on cpu; with cpu as default all land on cpu. Resets the
        # default device at the end.
        """Test that factory functions respect default device in compiled code"""

        @torch.compile(backend="eager", fullgraph=True)
        def random_func(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Creates random unsigned integers shaped like the input, a cpu
            # zeros tensor and a default-device zeros tensor like it; returns the input plus the
            # integers and the three tensors.
            # Test various factory functions
            rnd = torch.randint(0, 2**32, size=x.shape, dtype=torch.uint32)
            zeros = torch.zeros_like(rnd, device="cpu")
            zeros_matched = torch.zeros_like(rnd)
            return x + rnd, rnd, zeros, zeros_matched

        try:
            torch.set_default_device(device_type)
            (result, rnd, zeros, zeros_matched) = random_func(torch.randn(()))

            # Verify tensors are on the current accelerator
            self.assertEqual(rnd.device.type, device_type)
            self.assertEqual(result.device.type, device_type)
            self.assertEqual(zeros.device.type, "cpu")
            self.assertEqual(zeros_matched.device.type, rnd.device.type)

            torch.set_default_device("cpu")
            (result, rnd, zeros, zeros_matched) = random_func(torch.randn(()))

            # Verify tensors are on cpu
            self.assertEqual(rnd.device.type, "cpu")
            self.assertEqual(result.device.type, "cpu")
            self.assertEqual(zeros.device.type, "cpu")
            self.assertEqual(zeros_matched.device.type, rnd.device.type)
        finally:
            torch.set_default_device(None)

    @unittest.skipIf(not TEST_ACCELERATOR, "requires accelerator")
    def test_default_device_factory_functions_priority(self):
        # [FUNCTION DOC] Requires an accelerator. Checks that an explicit cpu device argument to
        # random integer creation overrides the accelerator default device in a compiled function,
        # while the sum result lands on the default device. Resets the default device at the end.
        try:
            torch.set_default_device(device_type)

            @torch.compile(backend="eager", fullgraph=True)
            def with_explicit_device(
                x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Creates random unsigned integers shaped like the input on
                # explicit cpu and returns the input plus them together with the integers.
                rnd = torch.randint(
                    0, 2**32, size=x.shape, dtype=torch.uint32, device="cpu"
                )
                return x + rnd, rnd

            (result, rnd) = with_explicit_device(torch.randn(()))
            self.assertEqual(rnd.device.type, "cpu")
            self.assertEqual(result.device.type, device_type)
        finally:
            torch.set_default_device(None)

    def test_torch_function_mode_no_leak_on_graph_break(self):
        # [FUNCTION DOC] Regression test: a mode used in a compiled function with an implicit
        # graph break must not stay on the C-level mode stack. The stack length after the call
        # must be zero.
        # Regression test for https://github.com/pytorch/pytorch/issues/182317
        # The resume function's except handler referenced a variable from
        # the first compiled graph's scope, causing a KeyError that masked
        # the original exception and left the mode on the C-level stack.
        @torch.compile(backend="eager")
        def fn():
            # [FUNCTION DOC] HELPER: Defines a mode that always returns minus one, creates a
            # tensor within it, and returns that tensor.
            class A(TorchFunctionMode):
                # [CLASS DOC] HELPER: Torch function mode returning minus one for everything.
                def __torch_function__(self, *args, **kwargs):
                    # [FUNCTION DOC] HELPER: Mode hook returning minus one regardless
                    # of arguments.
                    return -1

            with A():
                x = torch.tensor([1])
            return x

        fn()
        self.assertEqual(_len_torch_function_stack(), 0)

    def test_torch_function_mode_no_leak_nested_graph_break(self):
        # [FUNCTION DOC] Like the previous leak test but with an explicit graph break inside the
        # mode between two tensor creations; the stack length afterwards must be zero.
        @torch.compile(backend="eager")
        def fn():
            # [FUNCTION DOC] HELPER: Defines a constant mode, creates a tensor, graph breaks,
            # creates another tensor within it, and returns the second.
            class A(TorchFunctionMode):
                # [CLASS DOC] HELPER: Torch function mode returning minus one for everything.
                def __torch_function__(self, *args, **kwargs):
                    # [FUNCTION DOC] HELPER: Mode hook returning minus one regardless
                    # of arguments.
                    return -1

            with A():
                _x = torch.tensor([1])

                torch._dynamo.graph_break()

                y = torch.tensor([2])
            return y

        fn()
        self.assertEqual(_len_torch_function_stack(), 0)


class InvokeSubgraphBackendTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests of the invoke_subgraph backend: nested compiled regions emitted as higher
    # order invoke_subgraph nodes when tracing under make_fx or AOT export, subgraph sharing,
    # guard-failure splitting, forward and backward handling, input mutation and aliasing, and
    # flex attention with DTensor.
    @torch._dynamo.config.patch(
        trace_autograd_ops=True,
        inline_single_use_invoke_subgraph=False,
    )
    def test_eager_backend_traced_backward_with_compile_region(self):
        # [FUNCTION DOC] Compares a training step of a model whose blocks are marked as compile
        # regions against a deep copy run eagerly: loss and parameter gradients must match, the
        # recorded graph must have exactly one captured graph containing two invoke_subgraph
        # nodes (one per block). Uses tracing of autograd ops and no single-use inlining via
        # config patch.
        import copy

        from torch._higher_order_ops.invoke_subgraph import mark_compile_region

        @mark_compile_region
        def block_fwd(x, w1, b1, w2, b2):
            # [FUNCTION DOC] HELPER: Compile region body: relu of a linear transform then a
            # residual add of a second linear transform.
            h = torch.nn.functional.relu(torch.nn.functional.linear(x, w1, b1))
            return x + torch.nn.functional.linear(h, w2, b2)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a list of two sequential two-layer linear stacks.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds a module list of two sequential blocks, each with
                # a four by eight and an eight by four linear layer.
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(4, 8),
                            torch.nn.Linear(8, 4),
                        )
                        for _ in range(2)
                    ]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the compile region function to each block's
                # weights and biases in order and returns the final tensor.
                for blk in self.blocks:
                    fc1, fc2 = blk[0], blk[1]
                    x = block_fwd(x, fc1.weight, fc1.bias, fc2.weight, fc2.bias)
                return x

        torch.manual_seed(0)
        model = Model()
        ref_model = copy.deepcopy(model)
        x = torch.randn(4, 4)

        def run_step(m, inp):
            # [FUNCTION DOC] HELPER: Runs the model, sums the output as loss, runs backward and
            # returns the detached loss.
            loss = m(inp).sum()
            loss.backward()
            return loss.detach()

        ref_loss = run_step(ref_model, x.clone())

        def train_step(inp):
            # [FUNCTION DOC] HELPER: Runs one step on the compiled model with the given input.
            return run_step(model, inp)

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(train_step, backend=backend, fullgraph=True)
        loss = compiled(x.clone())

        self.assertEqual(loss, ref_loss)
        self.assertEqual(len(backend.graphs), 1)
        invoke_subgraph_nodes = [
            node
            for node in backend.graphs[0].graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.higher_order.invoke_subgraph
        ]
        self.assertEqual(len(invoke_subgraph_nodes), 2)
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            self.assertEqual(param.grad, ref_param.grad)

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_make_fx_over_compiled_function(self):
        # [FUNCTION DOC] Traces a function calling an invoke_subgraph backend compiled function
        # with a fake tensor make_fx and checks with an expected string that the graph has an
        # invoke_subgraph node and a repeated subgraph module instead of inlined ops. Resets
        # Dynamo first.
        """Test that make_fx can trace over torch.compile'd functions using invoke_subgraph backend.

        When force_compile_during_fx_trace=True, the invoke_subgraph backend should
        emit an invoke_subgraph HOP in the traced graph instead of inlining the subgraph.
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()  # Clear any cached graphs

        def simple_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the first input times two plus the second.
            return x * 2 + y

        compiled_fn = torch.compile(simple_fn, backend="invoke_subgraph")

        def outer_fn(x, y):
            # [FUNCTION DOC] HELPER: Adds one to the first input, calls the compiled function with
            # it and the second input, and doubles the result.
            z = x + 1
            result = compiled_fn(z, y)
            return result * 2

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # Trace with make_fx - the compiled_fn should appear as invoke_subgraph HOP
        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x, y)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = None
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', add, y_1);  repeated_subgraph0 = add = y_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(getitem, 2);  getitem = None
        return mul

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(mul, arg1_1);  mul = arg1_1 = None
            return (add,)
""",
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_same_compiled_fn_called_twice_shares_subgraph(self):
        # [FUNCTION DOC] Traces a function that calls the same compiled function twice and
        # checks via an expected string that both calls reference one shared subgraph with the
        # same identifier name. Runs inside a fresh tracing context so the cache works. Resets
        # Dynamo first.
        """Test that calling the same compiled function twice uses the same subgraph.

        When the same compiled function is called multiple times with inputs that
        don't cause guard failures, both calls should reference the same subgraph.
        """
        from torch._guards import tracing, TracingContext
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()

        def simple_fn(x):
            # [FUNCTION DOC] HELPER: Returns the input times two.
            return x * 2

        compiled_fn = torch.compile(simple_fn, backend="invoke_subgraph")

        def outer_fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the compiled function on each of two inputs and returns
            # the sum.
            # Call the same compiled function twice
            a = compiled_fn(x)
            b = compiled_fn(y)
            return a + b

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # Set up TracingContext so invoke_subgraph cache works
        tracing_ctx = TracingContext(fake_mode=None)
        with tracing(tracing_ctx):
            traced = make_fx(
                outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
            )(x, y)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', x_1);  repeated_subgraph0 = x_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'invoke_subgraph_0', y_1);  repeated_subgraph0_1 = y_1 = None
        getitem_1: "f32[3, 3]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return add

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None
            return (mul,)
""",
        )

    @torch._functorch.config.patch(guess_tangent_strides_as_outputs=True)
    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_invoke_subgraph_seq_nr(self):
        # [FUNCTION DOC] Checks sequence numbers on forward and backward graphs when an
        # invoke_subgraph backend function containing an annotated cosine is traced by an AOT
        # autograd function, with the forward and backward compilers capturing graphs. Compares
        # ordered sequence number groups of the main graphs and of the subgraphs against expected
        # lists, and the printed forward and backward graphs against expected strings ignoring
        # comments and empty lines.
        """
        Test that the seq_nr on the subgraphs and the invoke_subgraph HOP nodes are correct
        right before we copy metadata from fwd to bwd graph.
        """
        from torch._functorch.aot_autograd import aot_function
        from torch._guards import tracing, TracingContext

        torch._dynamo.reset()

        def inner_fn(x):
            # [FUNCTION DOC] HELPER: Under an annotation, computes cosine of the input, then
            # returns half of it.
            with torch.fx.traceback.annotate({"test": "test"}):
                y = x.cos()
            return y / 2

        compiled_fn = torch.compile(inner_fn, backend="invoke_subgraph")

        def outer_fn(x):
            # [FUNCTION DOC] HELPER: Adds one, calls the compiled function, and returns the sum.
            y = x + 1
            z = compiled_fn(y)
            return z.sum()

        x = torch.randn(3, 3, requires_grad=True)

        # Track forward graph to verify invoke_subgraph appears
        fw_graph = None
        bw_graph = None

        def fw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Forward compiler hook storing the graph module in the
            # enclosing variable and returning it.
            nonlocal fw_graph
            fw_graph = gm
            return gm

        def bw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backward compiler hook storing the graph module in the
            # enclosing variable and returning it.
            nonlocal bw_graph
            bw_graph = gm
            return gm

        # we expect to capture two graphs, the first one from aot_autograd in invoke_subgraph backend
        # This one should have correct seq_nr on the joint graph for inner_fn and copy the metadata.
        # the second one from actual aot_stage1_graph_capture.
        tracing_ctx = TracingContext(fake_mode=None)
        with tracing(tracing_ctx):
            aot_fn = aot_function(
                outer_fn,
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                _disable_torch_fn_metadata_mode=True,
            )

            # Run forward and backward
            result = aot_fn(x)
            result.backward()

        # Check seq_nr ordering for main forward and backward graphs
        main_groups = torch.fx.traceback._get_ordered_seq_nr_groups(
            [fw_graph, bw_graph]
        )
        self.assertEqual(
            main_groups,
            [
                ["add"],  # seq_nr 21
                [
                    "copy",
                    "empty_strided",
                    "getitem",
                    "getitem_1",
                    "getitem_2",
                    "invoke_subgraph",
                    "invoke_subgraph_1",
                ],  # seq_nr 22
                ["expand", "sum_1"],  # seq_nr 23
            ],
        )

        # Check seq_nr ordering for inner subgraphs (forward and backward)
        subgraph_groups = torch.fx.traceback._get_ordered_seq_nr_groups(
            [fw_graph.repeated_subgraph0, bw_graph.repeated_subgraph1]
        )
        self.assertEqual(
            subgraph_groups,
            [
                ["cos", "mul", "neg", "sin"],  # seq_nr 15
                ["div", "div"],  # seq_nr 16 - both forward and backward have div
            ],
        )

        # The annotation is not checked here because we used ignore_comments = True.
        # The comments here are helpful for human to read and understand the unit test.
        self.assertExpectedInline(
            normalize_gm(fw_graph.print_readable(print_output=False)),
            """
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]"):
        # Annotation: {'seq_nr': 13} No stacktrace found for following nodes
        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(primals_1, 1);  primals_1 = None

        # Annotation: {'seq_nr': 14} No stacktrace found for following nodes
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', add);  repeated_subgraph0 = add = None
        getitem: "f32[3, 3]" = invoke_subgraph[0]
        getitem_1: "f32[3, 3]" = invoke_subgraph[1];  invoke_subgraph = None

        # Annotation: {'seq_nr': 15} No stacktrace found for following nodes
        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None
        return (sum_1, getitem_1)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            # Annotation: {'test': 'test', 'seq_nr': 9} File: test_modes.py:920 in inner_fn, code: y = x.cos()
            cos: "f32[3, 3]" = torch.ops.aten.cos.default(arg0_1)

            # Annotation: {'seq_nr': 10} File: test_modes.py:921 in inner_fn, code: return y / 2
            div: "f32[3, 3]" = torch.ops.aten.div.Tensor(cos, 2);  cos = None
            return (div, arg0_1)
        """,
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        self.assertExpectedInline(
            normalize_gm(bw_graph.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, getitem_1: "f32[3, 3]", tangents_1: "f32[]"):
        expand: "f32[3, 3]" = torch.ops.aten.expand.default(tangents_1, [3, 3]);  tangents_1 = None
        empty_strided: "f32[3, 3]" = torch.ops.aten.empty_strided.default([3, 3], [3, 1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy: "f32[3, 3]" = torch.ops.aten.copy.default(empty_strided, expand);  empty_strided = expand = None
        repeated_subgraph1 = self.repeated_subgraph1
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph1, 'invoke_subgraph_1', getitem_1, copy);  repeated_subgraph1 = getitem_1 = copy = None
        getitem_2: "f32[3, 3]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_2,)
    class repeated_subgraph1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            div: "f32[3, 3]" = torch.ops.aten.div.Tensor(arg1_1, 2);  arg1_1 = None
            sin: "f32[3, 3]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            neg: "f32[3, 3]" = torch.ops.aten.neg.default(sin);  sin = None
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(div, neg);  div = neg = None
            return (mul,)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_guard_failure_creates_separate_subgraphs(self):
        # [FUNCTION DOC] Calls a compiled function with a true and then a false boolean flag
        # inside a make_fx trace; the guard failure must create a second subgraph with a
        # distinct identifier, checked by an expected string with two repeated subgraphs. Resets
        # Dynamo first.
        """Test that guard failures create separate subgraphs.

        When the same compiled function is called with inputs that cause guard
        failures (e.g., different bool values), each compilation should result
        in a separate invoke_subgraph with a different identifier.
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()

        def conditional_fn(x, flag: bool):
            # [FUNCTION DOC] HELPER: Returns the input times two if the flag is true, otherwise
            # times three.
            if flag:
                return x * 2
            else:
                return x * 3

        compiled_fn = torch.compile(conditional_fn, backend="invoke_subgraph")

        def outer_fn(x):
            # [FUNCTION DOC] HELPER: Calls the compiled function with the flag true and then false
            # and returns the sum.
            # Call with flag=True, then flag=False - should trigger recompilation
            a = compiled_fn(x, True)
            b = compiled_fn(x, False)
            return a + b

        x = torch.randn(3, 3)

        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', x_1);  repeated_subgraph0 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph1 = self.repeated_subgraph1
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph1, 'invoke_subgraph_1', x_1);  repeated_subgraph1 = x_1 = None
        getitem_1: "f32[3, 3]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return add

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None
            return (mul,)

    class repeated_subgraph1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, 3);  arg0_1 = None
            return (mul,)
""",
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_multiple_inputs(self):
        # [FUNCTION DOC] Traces a compiled function that takes four tensors, checking via an
        # expected string that one invoke_subgraph node receives all four inputs. Resets
        # Dynamo first.
        """Test invoke_subgraph with multiple tensor inputs.

        Verifies that the invoke_subgraph HOP correctly handles functions
        that take more than 2 tensor inputs.
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()

        def multi_input_fn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Returns first times second plus third times fourth.
            return a * b + c * d

        compiled_fn = torch.compile(multi_input_fn, backend="invoke_subgraph")

        def outer_fn(w, x, y, z):
            # [FUNCTION DOC] HELPER: Forwards four inputs to the compiled function.
            return compiled_fn(w, x, y, z)

        w = torch.randn(3, 3)
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.randn(3, 3)

        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(w, x, y, z)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(torch.nn.Module):
    def forward(self, w_1: "f32[3, 3]", x_1: "f32[3, 3]", y_1: "f32[3, 3]", z_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', w_1, x_1, y_1, z_1);  repeated_subgraph0 = w_1 = x_1 = y_1 = z_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        return getitem

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]", arg2_1: "f32[3, 3]", arg3_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            mul_1: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
            return (add,)
""",
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_multiple_outputs(self):
        # [FUNCTION DOC] Traces a compiled function that returns three tensors, checking via an
        # expected string that three values are extracted from the invoke_subgraph result. Resets
        # Dynamo first.
        """Test invoke_subgraph with multiple tensor outputs.

        Verifies that the invoke_subgraph HOP correctly handles functions
        that return multiple tensors.
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()

        def multi_output_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum, difference and product of the two inputs as
            # a tuple.
            return x + y, x - y, x * y

        compiled_fn = torch.compile(multi_output_fn, backend="invoke_subgraph")

        def outer_fn(a, b):
            # [FUNCTION DOC] HELPER: Unpacks the three outputs of the compiled function and
            # returns sum times difference plus product.
            sum_out, diff_out, prod_out = compiled_fn(a, b)
            return sum_out * diff_out + prod_out

        a = torch.randn(3, 3)
        b = torch.randn(3, 3)

        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(a, b)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(torch.nn.Module):
    def forward(self, a_1: "f32[3, 3]", b_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', a_1, b_1);  repeated_subgraph0 = a_1 = b_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0]
        getitem_1: "f32[3, 3]" = invoke_subgraph[1]
        getitem_2: "f32[3, 3]" = invoke_subgraph[2];  invoke_subgraph = None
        mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(mul, getitem_2);  mul = getitem_2 = None
        return add

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1)
            sub: "f32[3, 3]" = torch.ops.aten.sub.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            return (add, sub, mul)
""",
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_multiple_inputs_and_outputs(self):
        # [FUNCTION DOC] Traces a compiled function with three tensor inputs and two outputs,
        # checking via an expected string. Resets Dynamo first.
        """Test invoke_subgraph with both multiple inputs and outputs.

        Verifies that the invoke_subgraph HOP correctly handles functions
        that have both multiple tensor inputs and multiple tensor outputs.
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()

        def multi_io_fn(a, b, c):
            # [FUNCTION DOC] HELPER: Returns the sum of three inputs and the product of
            # three inputs.
            # Multiple inputs, multiple outputs
            return a + b + c, a * b * c

        compiled_fn = torch.compile(multi_io_fn, backend="invoke_subgraph")

        def outer_fn(x, y, z):
            # [FUNCTION DOC] HELPER: Calls the compiled function and returns the sum output minus
            # the product output.
            sum_out, prod_out = compiled_fn(x, y, z)
            return sum_out - prod_out

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.randn(3, 3)

        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x, y, z)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]", z_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', x_1, y_1, z_1);  repeated_subgraph0 = x_1 = y_1 = z_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0]
        getitem_1: "f32[3, 3]" = invoke_subgraph[1];  invoke_subgraph = None
        sub: "f32[3, 3]" = torch.ops.aten.sub.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return sub

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]", arg2_1: "f32[3, 3]"):
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
            add_1: "f32[3, 3]" = torch.ops.aten.add.Tensor(add, arg2_1);  add = None
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            mul_1: "f32[3, 3]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
            return (add_1, mul_1)
""",
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_aot_autograd_over_dynamo_with_requires_grad(self):
        # [FUNCTION DOC] Traces an AOT autograd function over a Dynamo compiled function with an
        # invoke_subgraph named counting backend, with inputs requiring gradient. Runs forward and
        # backward, asserts a forward graph was captured, that its code mentions invoke_subgraph,
        # and that Dynamo compiled exactly once.
        """Test AOTAutograd tracing over a torch.compile'd function with requires_grad inputs.

        This tests the scenario where:
        1. An outer aot_function traces a function with requires_grad inputs
        2. Inside that function, a torch.compile'd function with invoke_subgraph backend is called
        3. AOTAutograd partitions into forward/backward graphs
        4. The inner Dynamo region should only be compiled once
        """
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch._functorch.aot_autograd import aot_function
        from torch._functorch.compilers import nop

        torch._dynamo.reset()

        # Use a compile counter to track how many times Dynamo compiles
        compile_counter = CompileCounterWithBackend("invoke_subgraph")

        def inner_fn(x):
            # [FUNCTION DOC] HELPER: Returns the input times two plus one.
            return x * 2 + 1

        compiled_fn = torch.compile(inner_fn, backend=compile_counter)

        def outer_fn(x):
            # [FUNCTION DOC] HELPER: Adds one, calls the compiled function, and returns the sum.
            y = x + 1
            z = compiled_fn(y)
            return z.sum()

        x = torch.randn(3, 3, requires_grad=True)

        # Track forward graph to verify invoke_subgraph appears
        fw_graph = None

        def fw_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Forward compiler hook storing the graph module in the
            # enclosing variable and returning it.
            nonlocal fw_graph
            fw_graph = gm
            return gm

        aot_fn = aot_function(
            outer_fn,
            fw_compiler=fw_compiler,
            bw_compiler=nop,
            _disable_torch_fn_metadata_mode=True,
        )

        # Run forward and backward
        result = aot_fn(x)
        result.backward()

        # Check that we got a forward graph with invoke_subgraph
        self.assertIsNotNone(fw_graph, "Expected a forward graph")
        fw_graph_code = fw_graph.print_readable(print_output=False)
        self.assertIn("invoke_subgraph", fw_graph_code)

        # Check compile count - should be 1 (compiled once during tracing)
        self.assertEqual(
            compile_counter.frame_count,
            1,
            lambda msg: f"{msg}\nExpected 1 compilation, got {compile_counter.frame_count}",
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_aot_autograd_over_dynamo_train_step(self):
        # [FUNCTION DOC] Traces a full functional training step with make_fx using a parameter
        # tuple taken from a small two-layer model, a compiled middle function, and autograd grad.
        # Runs the traced graph, checks loss is scalar, four gradients have matching parameter
        # shapes, the graph code mentions invoke_subgraph, and Dynamo compiled once.
        """Test a full training step with nn.Module traced by AOTAutograd over Dynamo.

        This tests a realistic training scenario where:
        1. We have an actual nn.Module with parameters
        2. make_fx traces the forward and backward as a single graph
        3. Inside the module, a torch.compile'd function with invoke_subgraph is called
        4. We use torch.autograd.grad to compute gradients (not .backward())
        5. Gradients are returned explicitly from the traced function
        """
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()

        # Use a compile counter to track how many times Dynamo compiles
        compile_counter = CompileCounterWithBackend("invoke_subgraph")

        # Define a simple MLP module
        class SimpleMLP(torch.nn.Module):
            # [CLASS DOC] HELPER: Two-layer perceptron with a four to eight and an eight to four
            # linear layer and relu in between. Only its parameters are used.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two linear layers.
                super().__init__()
                self.fc1 = torch.nn.Linear(4, 8)
                self.fc2 = torch.nn.Linear(8, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first layer, relu, then the second layer.
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x

        model = SimpleMLP()

        # Inner compiled function that will be called inside the training step
        def inner_fn(x):
            # [FUNCTION DOC] HELPER: Returns the input times two plus one.
            return x * 2 + 1

        compiled_inner = torch.compile(inner_fn, backend=compile_counter)

        # Training step function that uses the model and compiled inner function
        def train_step(params, x):
            # [FUNCTION DOC] HELPER: Applies functional linear, relu, the compiled function,
            # and a second linear to the input using the four parameters, sums to a loss,
            # computes gradients with respect to all parameters, and returns the loss followed
            # by the gradients.
            # params is a flat list of parameters: [fc1.weight, fc1.bias, fc2.weight, fc2.bias]
            fc1_weight, fc1_bias, fc2_weight, fc2_bias = params

            # Manual forward pass using functional style
            h = torch.nn.functional.linear(x, fc1_weight, fc1_bias)
            h = torch.relu(h)
            # Apply compiled inner function
            h = compiled_inner(h)
            out = torch.nn.functional.linear(h, fc2_weight, fc2_bias)

            # Compute loss
            loss = out.sum()

            # Compute gradients using torch.autograd.grad
            grads = torch.autograd.grad(loss, params)

            return (loss, *grads)

        # Prepare inputs
        x = torch.randn(2, 4)
        params = (
            model.fc1.weight.detach().clone().requires_grad_(True),
            model.fc1.bias.detach().clone().requires_grad_(True),
            model.fc2.weight.detach().clone().requires_grad_(True),
            model.fc2.bias.detach().clone().requires_grad_(True),
        )

        # Trace using make_fx
        fx_graph = make_fx(train_step, _disable_torch_fn_metadata_mode=True)(params, x)

        # Run the traced graph
        result = fx_graph(params, x)
        loss = result[0]
        grads = result[1:]

        # Basic sanity checks
        self.assertEqual(loss.shape, ())
        self.assertEqual(len(grads), 4)
        self.assertEqual(grads[0].shape, params[0].shape)  # fc1.weight grad
        self.assertEqual(grads[1].shape, params[1].shape)  # fc1.bias grad
        self.assertEqual(grads[2].shape, params[2].shape)  # fc2.weight grad
        self.assertEqual(grads[3].shape, params[3].shape)  # fc2.bias grad

        # Check that we got a graph with invoke_subgraph
        graph_code = fx_graph.print_readable(print_output=False)
        self.assertIn("invoke_subgraph", graph_code)

        # Check compile count - should be exactly 1 compilation
        self.assertEqual(
            compile_counter.frame_count,
            1,
            lambda msg: f"{msg}\nExpected 1 compilation, got {compile_counter.frame_count}",
        )

    @unittest.skipIf(not HAS_GPU, "requires GPU and Triton")
    def test_nested_compile_dynamic(self):
        # [FUNCTION DOC] Requires a GPU and Triton. Compiles a linear layer with inductor, dynamic
        # shapes and the wrap compiled regions option, then runs batch sizes two and five,
        # checking shapes and closeness to eager.
        # KNOWN ISSUE: the docstring says wrap_compiled_regions but the option set
        # is wrap_inductor_compiled_regions.
        """Test that wrap_compiled_regions works with dynamic shapes."""

        d_model = 64

        class MMLayer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer of the given width.
            def __init__(self, d_model: int):
                # [FUNCTION DOC] HELPER: Creates a square linear layer of the given width.
                super().__init__()
                self.linear = torch.nn.Linear(d_model, d_model)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.linear(x)

        torch._dynamo.reset()

        layer = MMLayer(d_model).to(device_type)
        compiled_mm = torch.compile(
            layer,
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            dynamic=True,
        )

        x = torch.randn(2, d_model, device=device_type)
        result = compiled_mm(x)
        self.assertEqual(result.shape, (2, d_model))
        torch.testing.assert_close(result, layer(x))

        # Different batch size reuses the same compiled code
        x2 = torch.randn(5, d_model, device=device_type)
        result2 = compiled_mm(x2)
        self.assertEqual(result2.shape, (5, d_model))
        torch.testing.assert_close(result2, layer(x2))

    @unittest.skipIf(not HAS_GPU, "requires GPU and Triton")
    def test_nested_compile_input_mutation(self):
        # [FUNCTION DOC] Requires GPU and Triton. Exports through AOT a stack of two compiled
        # layers, each mutating a buffer inside an annotated region, using a regional inductor
        # plus invoke_subgraph inner compiler as the backend, with fake tensor tracing and
        # several config patches. Compiles the joint graph with no-op compilers, runs it with
        # saved parameters and buffers, and calls backward. Prints the outer graph; passes if
        # nothing raises.
        """Test nested compile with input mutation inside a compiled region.

        Uses regional_inductor with fx_traceback.annotate to compile
        a layer that mutates a buffer.
        """
        import contextlib

        import torch.fx.traceback as fx_traceback
        from functorch.compile import nop
        from torch._dynamo.backends.common import aot_autograd
        from torch._dynamo.backends.debugging import invoke_subgraph_inner_compiler
        from torch._export.utils import _compiling_state_context
        from torch._functorch.aot_autograd import (
            aot_compile_joint_with_descriptors,
            aot_export_joint_with_descriptors,
        )
        from torch._guards import tracing, TracingContext
        from torch._subclasses import FakeTensorMode
        from torch.fx.passes.regional_inductor import regional_inductor

        def regional_inductor_invoke_subgraph(gm, args):
            # [FUNCTION DOC] HELPER: Applies the regional inductor to the graph module and passes
            # the result to the invoke_subgraph inner compiler, returning that.
            out_gm = regional_inductor(gm, args)
            return invoke_subgraph_inner_compiler(out_gm, args)

        def aot_eager_regional_inductor_invoke_subgraph():
            # [FUNCTION DOC] HELPER: Builds an AOT autograd backend whose forward and backward
            # compilers are the regional-inductor helper, keeping input mutations.
            return aot_autograd(
                fw_compiler=regional_inductor_invoke_subgraph,
                bw_compiler=regional_inductor_invoke_subgraph,
                keep_inference_input_mutations=True,
            )

        d_model = 64

        class MutatingLayer(torch.nn.Module):
            # [CLASS DOC] HELPER: Layer with a linear module and an int64 call counter buffer
            # which is incremented in an annotated region marked for inductor compile.
            def __init__(self, d_model: int):
                # [FUNCTION DOC] HELPER: Creates the linear layer and registers the zero
                # counter buffer.
                super().__init__()
                self.linear = torch.nn.Linear(d_model, d_model)
                self.register_buffer("call_count", torch.tensor(0, dtype=torch.int64))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Increments the counter in place under an annotation and
                # applies the linear layer.
                with fx_traceback.annotate({"compile_with_inductor": 0}):
                    self.call_count.add_(1)
                return self.linear(x)

        class StackedMutating(torch.nn.Module):
            # [CLASS DOC] HELPER: Stack of compiled mutating layers, each compiled with the
            # regional inductor invoke_subgraph backend.
            def __init__(self, d_model: int, n_layers: int = 2):
                # [FUNCTION DOC] HELPER: Builds a module list of compiled mutating layers,
                # n_layers many.
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [
                        torch.compile(
                            MutatingLayer(d_model),
                            backend=aot_eager_regional_inductor_invoke_subgraph(),
                        )
                        for _ in range(n_layers)
                    ]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each layer in order.
                for layer in self.layers:
                    x = layer(x)
                return x

        with (
            torch._dynamo.config.patch(force_compile_during_fx_trace=True),
            torch._inductor.config.patch(wrap_inductor_compiled_regions=True),
            torch._functorch.config.patch(force_non_lazy_backward_lowering=True),
        ):
            torch._dynamo.reset()

            model = StackedMutating(d_model, n_layers=2).to(device_type)

            x = torch.randn(2, d_model, device=device_type, requires_grad=True)

            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            saved_params = list(model.parameters())
            saved_buffers = list(model.buffers())

            with contextlib.ExitStack() as stack:
                stack.enter_context(tracing(TracingContext(fake_mode)))
                stack.enter_context(_compiling_state_context())
                stack.enter_context(fake_mode)

                joint_with_descriptors = aot_export_joint_with_descriptors(
                    stack,
                    model,
                    args=(x,),
                    kwargs={},
                    keep_inference_input_mutations=True,
                    _disable_torch_fn_metadata_mode=True,
                )
                gm = joint_with_descriptors.graph_module
                print("=== input_mutation outer graph ===")
                print(gm.print_readable(print_output=False))
                compiled_fn = aot_compile_joint_with_descriptors(
                    joint_with_descriptors,
                    fw_compiler=nop,
                    bw_compiler=nop,
                )

            def wrapped_fn(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls the compiled joint function with the saved
                # parameters and buffers followed by the given arguments.
                return compiled_fn(
                    *saved_params,
                    *saved_buffers,
                    *args,
                    **kwargs,
                )

            out = wrapped_fn(x)
            out.sum().backward()

    @unittest.skipIf(not HAS_GPU, "requires GPU and Triton")
    def test_nested_compile_output_aliases_input(self):
        # [FUNCTION DOC] Requires GPU and Triton. Like the input mutation test but each layer
        # returns a view of a linear output (unsqueeze then squeeze under annotation) so the
        # output aliases intermediate data. Exports, compiles with no-op compilers, runs and
        # calls backward.
        """Test nested compile where output is a view-alias of input.

        Uses regional_inductor with fx_traceback.annotate to compile
        a layer that returns a view of its input.
        """
        import contextlib

        import torch.fx.traceback as fx_traceback
        from functorch.compile import nop
        from torch._dynamo.backends.common import aot_autograd
        from torch._dynamo.backends.debugging import invoke_subgraph_inner_compiler
        from torch._export.utils import _compiling_state_context
        from torch._functorch.aot_autograd import (
            aot_compile_joint_with_descriptors,
            aot_export_joint_with_descriptors,
        )
        from torch._guards import tracing, TracingContext
        from torch._subclasses import FakeTensorMode
        from torch.fx.passes.regional_inductor import regional_inductor

        def regional_inductor_invoke_subgraph(gm, args):
            # [FUNCTION DOC] HELPER: Applies the regional inductor and passes the result to the
            # invoke_subgraph inner compiler.
            out_gm = regional_inductor(gm, args)
            return invoke_subgraph_inner_compiler(out_gm, args)

        def aot_eager_regional_inductor_invoke_subgraph():
            # [FUNCTION DOC] HELPER: Builds an AOT autograd backend using the regional-inductor
            # helper for forward and backward.
            return aot_autograd(
                fw_compiler=regional_inductor_invoke_subgraph,
                bw_compiler=regional_inductor_invoke_subgraph,
                keep_inference_input_mutations=True,
            )

        d_model = 64

        class ViewLayer(torch.nn.Module):
            # [CLASS DOC] HELPER: Layer applying a linear layer then returning a view of it
            # through an annotated unsqueeze and squeeze.
            def __init__(self, d_model: int):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(d_model, d_model)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear, then returns the unsqueezed then squeezed
                # view under an annotation.
                y = self.linear(x)
                with fx_traceback.annotate({"compile_with_inductor": 0}):
                    # View-alias: unsqueeze then squeeze back
                    return y.unsqueeze(1).squeeze(1)

        class StackedView(torch.nn.Module):
            # [CLASS DOC] HELPER: Stack of compiled view layers.
            def __init__(self, d_model: int, n_layers: int = 2):
                # [FUNCTION DOC] HELPER: Builds a module list of compiled view layers.
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [
                        torch.compile(
                            ViewLayer(d_model),
                            backend=aot_eager_regional_inductor_invoke_subgraph(),
                        )
                        for _ in range(n_layers)
                    ]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each layer in order.
                for layer in self.layers:
                    x = layer(x)
                return x

        with (
            torch._dynamo.config.patch(force_compile_during_fx_trace=True),
            torch._inductor.config.patch(wrap_inductor_compiled_regions=True),
            torch._functorch.config.patch(force_non_lazy_backward_lowering=True),
        ):
            torch._dynamo.reset()

            model = StackedView(d_model, n_layers=2).to(device_type)

            x = torch.randn(2, d_model, device=device_type, requires_grad=True)

            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            saved_params = list(model.parameters())
            saved_buffers = list(model.buffers())

            with contextlib.ExitStack() as stack:
                stack.enter_context(tracing(TracingContext(fake_mode)))
                stack.enter_context(_compiling_state_context())
                stack.enter_context(fake_mode)

                joint_with_descriptors = aot_export_joint_with_descriptors(
                    stack,
                    model,
                    args=(x,),
                    kwargs={},
                    keep_inference_input_mutations=True,
                    _disable_torch_fn_metadata_mode=True,
                )
                gm = joint_with_descriptors.graph_module
                print("=== output_aliases_input outer graph ===")
                print(gm.print_readable(print_output=False))
                compiled_fn = aot_compile_joint_with_descriptors(
                    joint_with_descriptors,
                    fw_compiler=nop,
                    bw_compiler=nop,
                )

            def wrapped_fn(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls the compiled joint function with saved parameters
                # and buffers followed by the given arguments.
                return compiled_fn(
                    *saved_params,
                    *saved_buffers,
                    *args,
                    **kwargs,
                )

            out = wrapped_fn(x)
            out.sum().backward()

    @unittest.skipUnless(
        IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED and not torch.version.hip,
        "Requires CUDA with SM >= 8.0, Triton, and not ROCm",
    )
    def test_nested_compile_transformer_with_flex_attention_compiled_layers(
        self,
    ):
        # [FUNCTION DOC] Requires CUDA with flex attention support and not ROCm. Builds a
        # four-layer transformer with flex attention and a block mask whose mask function closes
        # over a tensor. Each layer is compiled with the regional inductor invoke_subgraph backend
        # and the model's parameters are replicated DTensors on a fake process group. Exports the
        # joint graph, partitions it, compiles with no-op compilers, runs and calls backward, then
        # destroys the process group.
        # KNOWN ISSUE: the partition output graphs are unused, and no check on invoke_subgraph
        # nodes is made although the docstring promises verification.
        """Test a transformer model with 4 identical compiled layers using flex_attention.

        This test:
        1. Creates a small transformer model with 4 identical layers using flex_attention
        2. Compiles each layer with the invoke_subgraph + regional_inductor backend
        3. Passes the entire model through aot_export_joint_with_descriptors()
        4. Verifies the final graph has invoke_subgraph HOPs
        """
        import torch.distributed as dist
        import torch.fx.traceback as fx_traceback
        from torch._dynamo.backends.common import aot_autograd
        from torch._dynamo.backends.debugging import invoke_subgraph_inner_compiler
        from torch._guards import tracing, TracingContext
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import (
            distribute_module,
            distribute_tensor,
            DTensor,
            Replicate,
        )
        from torch.fx.passes.regional_inductor import regional_inductor
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        from torch.testing._internal.distributed.fake_pg import FakeStore

        def regional_inductor_invoke_subgraph(gm, args):
            # [FUNCTION DOC] HELPER: Applies the regional inductor and passes the result to the
            # invoke_subgraph inner compiler.
            out_gm = regional_inductor(gm, args)
            return invoke_subgraph_inner_compiler(out_gm, args)

        def aot_eager_regional_inductor_invoke_subgraph():
            # [FUNCTION DOC] HELPER: Builds an AOT autograd backend using the regional-inductor
            # helper for forward and backward.
            return aot_autograd(
                fw_compiler=regional_inductor_invoke_subgraph,
                bw_compiler=regional_inductor_invoke_subgraph,
                # Keep input mutations in the graph
                keep_inference_input_mutations=True,
            )

        d_model, n_heads, d_ff = 64, 4, 128
        batch_size, seq_len = 2, 32

        # dumb mask mod that closes over a tensor
        mask_bias = torch.tensor(0, device=device_type, dtype=torch.int32)

        def mask_mod(b_idx, h_idx, q_idx, k_idx):
            # [FUNCTION DOC] HELPER: Mask function allowing the key when the query index is not
            # before it, or when the enclosing bias tensor equals one.
            return (q_idx >= k_idx) | (mask_bias == 1)

        # Create block mask from the mask_mod
        block_mask = create_block_mask(
            mask_mod=mask_mod,
            B=batch_size,
            H=None,  # Broadcast over heads
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device_type,
        )

        # Transformer layer with flex_attention
        class TransformerLayer(torch.nn.Module):
            # [CLASS DOC] HELPER: Pre-norm transformer layer with four linear projections, two
            # layer norms, a feed-forward block, and an int64 call count buffer, using flex
            # attention on local shards and rewrapping to a DTensor.
            def __init__(self, d_model: int, n_heads: int, d_ff: int):
                # [FUNCTION DOC] HELPER: Stores head count and dimensions and creates projections,
                # norms, feed-forward block and the counter buffer.
                super().__init__()
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                self.d_model = d_model

                # Attention projections
                self.q_proj = torch.nn.Linear(d_model, d_model)
                self.k_proj = torch.nn.Linear(d_model, d_model)
                self.v_proj = torch.nn.Linear(d_model, d_model)
                self.out_proj = torch.nn.Linear(d_model, d_model)

                # Layer norms
                self.norm1 = torch.nn.LayerNorm(d_model)
                self.norm2 = torch.nn.LayerNorm(d_model)

                # FFN
                self.ff = torch.nn.Sequential(
                    torch.nn.Linear(d_model, d_ff),
                    torch.nn.GELU(),
                    torch.nn.Linear(d_ff, d_model),
                )

                # give the model a buffer mutation so we can test auto_functionalize too
                self.register_buffer("_call_count", torch.tensor(0, dtype=torch.int64))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Increments the buffer counter, normalizes, projects to
                # query, key and value, reshapes to heads, runs flex attention on local shards
                # under an annotation, rewraps as a replicated DTensor, merges heads, projects
                # out, adds the residual and the feed-forward residual, and returns the result.
                # Mutate the buffer to trigger input mutation detection
                self._call_count.add_(1)

                batch_size, seq_len, _ = x.shape

                # Pre-norm for attention
                normed = self.norm1(x)

                # Compute Q, K, V
                q = self.q_proj(normed)
                k = self.k_proj(normed)
                v = self.v_proj(normed)

                # Reshape for multi-head attention: (B, S, D) -> (B, H, S, D_head)
                q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(
                    1, 2
                )
                k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(
                    1, 2
                )
                v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(
                    1, 2
                )

                with fx_traceback.annotate({"compile_with_inductor": 0}):
                    attn_out_local = flex_attention(
                        q.to_local(),
                        k.to_local(),
                        v.to_local(),
                        block_mask=block_mask,
                    )
                    attn_out = DTensor.from_local(
                        attn_out_local, device_mesh, [Replicate()]
                    )

                attn_out = (
                    attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                )
                attn_out = self.out_proj(attn_out)

                x = x + attn_out

                x = x + self.ff(self.norm2(x))

                return x

        # Transformer model with 4 identical layers
        class SmallTransformer(torch.nn.Module):
            # [CLASS DOC] HELPER: Stack of compiled transformer layers.
            def __init__(
                self, d_model: int, n_heads: int, d_ff: int, n_layers: int = 4
            ):
                # [FUNCTION DOC] HELPER: Builds a module list of n_layers compiled
                # transformer layers.
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [
                        torch.compile(
                            TransformerLayer(d_model, n_heads, d_ff),
                            backend=aot_eager_regional_inductor_invoke_subgraph(),
                        )
                        for _ in range(n_layers)
                    ]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each layer in order.
                for layer in self.layers:
                    x = layer(x)
                return x

        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=0, world_size=2)
        device_mesh = init_device_mesh(device_type, (2,))

        with (
            # Needed when wrapping a compiled region with FX tracing
            torch._dynamo.config.patch(force_compile_during_fx_trace=True),
            # Needed because our inner compiled region uses inductor (for flex)
            torch._inductor.config.patch(wrap_inductor_compiled_regions=True),
            # AOTAutograd normally tries to "delay backward compilation to bw runtime",
            # but for nested compile we actually need it to happen when we trace the fw.
            torch._functorch.config.patch(force_non_lazy_backward_lowering=True),
        ):
            torch._dynamo.reset()

            model = SmallTransformer(d_model, n_heads, d_ff, n_layers=4)
            model = model.to(device_type)

            def replicate_all(name, module, device_mesh):
                # [FUNCTION DOC] HELPER: Partition function for distribute_module; replaces each
                # direct parameter of the module with a replicated distributed parameter on the
                # device mesh.
                for param_name, param in module.named_parameters(recurse=False):
                    dist_param = torch.nn.Parameter(
                        distribute_tensor(param, device_mesh, [Replicate()])
                    )
                    module.register_parameter(param_name, dist_param)

            model = distribute_module(model, device_mesh, replicate_all)

            x = torch.randn(
                batch_size,
                seq_len,
                d_model,
                device=device_type,
                dtype=torch.float32,
                requires_grad=True,
            )
            x = DTensor.from_local(x, device_mesh, [Replicate()])

            import contextlib

            from functorch.compile import nop
            from torch._export.utils import _compiling_state_context
            from torch._functorch.aot_autograd import (
                aot_compile_joint_with_descriptors,
                aot_export_joint_with_descriptors,
            )
            from torch._functorch.partitioners import default_partition
            from torch._guards import detect_fake_mode
            from torch._subclasses import FakeTensorMode

            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

            saved_params = list(model.parameters())
            saved_buffers = list(model.buffers())

            with contextlib.ExitStack() as stack:
                stack.enter_context(tracing(TracingContext(fake_mode)))
                stack.enter_context(_compiling_state_context())
                if fake_mode is not None:
                    stack.enter_context(fake_mode)

                joint_with_descriptors = aot_export_joint_with_descriptors(
                    stack,
                    model,
                    args=(x,),
                    kwargs={},
                    keep_inference_input_mutations=True,
                    # see https://github.com/pytorch/pytorch/pull/172087
                    _disable_torch_fn_metadata_mode=True,
                )
                gm = joint_with_descriptors.graph_module
                fake_mode = detect_fake_mode()
                num_fwd_outputs = (
                    joint_with_descriptors._aot_state.fw_metadata.num_forward
                )
                fwd_gm, bwd_gm = default_partition(
                    gm, None, num_fwd_outputs=num_fwd_outputs
                )
                compiled_fn = aot_compile_joint_with_descriptors(
                    joint_with_descriptors,
                    fw_compiler=nop,
                    bw_compiler=nop,
                )

            def wrapped_fn(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls the compiled joint function with saved parameters
                # and buffers followed by the given arguments.
                return compiled_fn(
                    *saved_params,
                    *saved_buffers,
                    *args,
                    **kwargs,
                )

            out = wrapped_fn(x)
            out.sum().backward()

        dist.destroy_process_group()

    @unittest.skipUnless(
        IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED and not torch.version.hip,
        "Requires CUDA with SM >= 8.0, Triton, and not ROCm",
    )
    def test_2tier_blockmask_tensor_closure_nested_compile_aot_export(self):
        # [FUNCTION DOC] Requires CUDA with flex attention support and not ROCm. Registers
        # BlockMask as a pytree node if not already registered, then AOT exports a model that
        # creates a block mask from a mask function capturing sliced tensors of an input and
        # returns the mask beside the output. Prints the graph and each submodule graph; passes if
        # export does not raise.
        """2-tier AOT export with BlockMask whose mask_mod captures tensors.

        Reproduces the sixlib/mango pattern where:
        - BlockMask is pytree-registered (as in sixlib/attention_mask.py)
        - BlockMask with tensor closure is created inside forward
        - aot_export flattens intermediates via pytree, exposing the
          mask_mod callable (with captured tensors) in the context

        visibility is a model input -> FunctionalTensor during outer trace.
        mask_mod closure captures derived slices (lower, upper).
        """
        import contextlib

        import torch.fx.traceback as fx_traceback
        from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
        from torch._subclasses import FakeTensorMode
        from torch.nn.attention.flex_attention import (
            BlockMask,
            create_block_mask,
            flex_attention,
        )
        from torch.utils._pytree import register_pytree_node, SUPPORTED_NODES

        # Register BlockMask as pytree node (same as sixlib/attention_mask.py)
        if BlockMask not in SUPPORTED_NODES:
            register_pytree_node(
                BlockMask,
                BlockMask._flatten,
                BlockMask._unflatten,
                flatten_with_keys_fn=BlockMask._flatten_with_keys,
                serialized_type_name="torch.nn.attention.flex_attention.BlockMask",
            )

        d_model, n_heads = 64, 4
        batch_size, seq_len = 2, 32

        class SimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with three linear projections whose forward builds a block
            # mask closing over slices of the visibility input and runs flex attention.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the three linear projections.
                super().__init__()
                self.q_proj = torch.nn.Linear(d_model, d_model)
                self.k_proj = torch.nn.Linear(d_model, d_model)
                self.v_proj = torch.nn.Linear(d_model, d_model)

            def forward(self, x, visibility):
                # [FUNCTION DOC] HELPER: Slices lower and upper bounds from the visibility input,
                # builds a block mask, projects query, key and value into heads, runs flex
                # attention under an annotation, merges heads and returns the output together with
                # the block mask.
                bs, sl, _ = x.shape
                lower = visibility[:, 0, :]
                upper = visibility[:, 1, :]

                def mask_mod(b_idx, h_idx, q_idx, k_idx):
                    # [FUNCTION DOC] HELPER: Mask function true when the key index is at least the
                    # lower bound and at most the upper bound for the batch and query position.
                    return (lower[b_idx, q_idx] <= k_idx) & (
                        k_idx <= upper[b_idx, q_idx]
                    )

                bm = create_block_mask(
                    mask_mod,
                    B=bs,
                    H=None,
                    Q_LEN=sl,
                    KV_LEN=sl,
                    device=x.device,
                )

                q = (
                    self.q_proj(x)
                    .view(bs, sl, n_heads, d_model // n_heads)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(bs, sl, n_heads, d_model // n_heads)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(bs, sl, n_heads, d_model // n_heads)
                    .transpose(1, 2)
                )

                with fx_traceback.annotate({"compile_with_inductor": 1}):
                    attn_out = flex_attention(q, k, v, block_mask=bm)

                out = attn_out.transpose(1, 2).contiguous().view(bs, sl, -1)
                # Return BlockMask alongside output so it appears in the
                # output pytree spec — this matches how sixlib's decoder
                # returns context alongside outputs.
                return out, bm

        with (
            torch._dynamo.config.patch(force_compile_during_fx_trace=True),
            torch._inductor.config.patch(wrap_inductor_compiled_regions=True),
            torch._functorch.config.patch(force_non_lazy_backward_lowering=True),
        ):
            torch._dynamo.reset()

            model = SimpleModel().to(device_type)
            x = torch.randn(
                batch_size,
                seq_len,
                d_model,
                device=device_type,
                dtype=torch.float32,
                requires_grad=True,
            )
            visibility = torch.zeros(
                batch_size,
                2,
                seq_len,
                device=device_type,
                dtype=torch.int64,
            )
            visibility[:, 1, :] = seq_len - 1

            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            with contextlib.ExitStack() as stack:
                stack.enter_context(fake_mode)
                result = aot_export_joint_with_descriptors(
                    stack,
                    model,
                    args=(x, visibility),
                    kwargs={},
                    keep_inference_input_mutations=True,
                    _disable_torch_fn_metadata_mode=True,
                )

            print("=== Outer (aot_export) graph ===")
            print(result.graph_module.graph)
            for name, submod in result.graph_module.named_modules():
                if name and hasattr(submod, "graph"):
                    print(f"\n=== Submodule: {name} ===")
                    print(submod.graph)


class TorchFunctionModeLifecycleTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Lifecycle test checking the default device is not leaked by the mode tests.
    def test_default_device_restored_after_mode_tests(self):
        # [FUNCTION DOC] Manually runs class setup, per-test setup, the stack mutation default
        # device test, per-test teardown and class teardown of the torch function mode test class,
        # then asserts no default device context remains in the current mode stack.
        case = TorchFunctionModeTests("test_stack_state_mutation_default_device")
        TorchFunctionModeTests.setUpClass()
        try:
            case.setUp()
            try:
                case.test_stack_state_mutation_default_device()
            finally:
                case.tearDown()
        finally:
            TorchFunctionModeTests.tearDownClass()

        stack = _get_current_function_mode_stack()
        self.assertFalse(any(isinstance(mode, DeviceContext) for mode in stack))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
