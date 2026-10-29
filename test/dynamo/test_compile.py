# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for torch.compile and Module.compile behavior: in-place module compilation,
# saving and loading compiled modules, compile callbacks, error propagation for invalid code,
# TorchScript handling, signature parity between torch.compiler public functions and
# torch._dynamo, and fullgraph errors when frames are skipped. Defines a small ToyModel at
# import time.

import inspect
import io
import os
import tempfile
from unittest.mock import patch

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import munge_exc


class ToyModel(torch.nn.Module):
# [CLASS DOC] HELPER: Small module with a Linear(10, 10) followed by a ReLU, used as the test
# subject.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Create the linear and relu submodules.
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Return relu applied to the linear output.
        return self.relu(self.linear(x))


class InPlaceCompilationTests(TestCase):
# [CLASS DOC] Test case for compiling modules in place and related compile-time behavior.
    def test_compilation(self):
        # [FUNCTION DOC] Reset dynamo, compile a ToyModel in place with a counting backend, run
        # it once and assert exactly one frame was compiled.
        torch._dynamo.reset()
        model = ToyModel()
        cnt = CompileCounter()
        model.compile(backend=cnt)
        x = torch.randn(10, 10)
        model(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_compilation_builtin_leaf_module(self):
        # [FUNCTION DOC] Compile a Conv2d in place with a counting backend; assert the output
        # equals the eager reference and that one frame and one graph op were counted.
        torch._dynamo.reset()
        model = torch.nn.Conv2d(3, 3, 3)
        cnt = CompileCounter()
        x = torch.randn(2, 3, 8, 8)
        ref = model(x)

        model.compile(backend=cnt)
        self.assertEqual(model(x), ref)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_overwrite_call_impl(self):
        # [FUNCTION DOC] Assert the module's compiled call implementation attribute is unset
        # before calling compile and set afterwards.
        torch._dynamo.reset()
        model = ToyModel()
        self.assertTrue(model._compiled_call_impl is None)
        model.compile()
        self.assertTrue(model._compiled_call_impl is not None)

    def test_save(self):
        # [FUNCTION DOC] Compile and run a module, torch.save the whole module to a temporary
        # directory, load it back with weights_only False and run it.
        torch._dynamo.reset()
        model = ToyModel()
        model.compile()
        model(torch.randn(1, 10))

        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(model, os.path.join(tmpdirname, "model.pt"))
            # weights_only=False as this is a legacy use case that loads a module
            loaded_model = torch.load(
                os.path.join(tmpdirname, "model.pt"), weights_only=False
            )
            loaded_model(torch.randn(1, 10))

    def test_state_dict_save(self):
        # [FUNCTION DOC] Compile and run a module, save only its state_dict to a temporary
        # directory, load it into a fresh ToyModel and run that.
        torch._dynamo.reset()
        model = ToyModel()
        model.compile()
        model(torch.randn(1, 10))
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(model.state_dict(), os.path.join(tmpdirname, "model.pt"))
            loaded_model = ToyModel()
            loaded_model.load_state_dict(
                # weights_only=False as this is a legacy use case that loads a module
                torch.load(os.path.join(tmpdirname, "model.pt"), weights_only=False)
            )
            loaded_model(torch.randn(1, 10))

    def test_jit_save(self):
        # [FUNCTION DOC] Compile and run a module, script it with torch.jit.script, save and
        # load the scripted module via torch.jit and run the loaded one.
        torch._dynamo.reset()
        model = ToyModel()
        model.compile()
        model(torch.randn(1, 10))
        scripted_model = torch.jit.script(model)
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.jit.save(scripted_model, os.path.join(tmpdirname, "model.pt"))
            loaded_model = torch.jit.load(os.path.join(tmpdirname, "model.pt"))
            loaded_model(torch.randn(1, 10))

    def test_compilation_callback(self):
        # [FUNCTION DOC] Register start and end compile callbacks that print messages; compile a
        # module with eager fullgraph, run it with stdout captured and assert the output is the
        # start message then the end message.
        torch._dynamo.reset()

        @torch._dynamo.on_compile_start
        def start_callback(_):
            # [FUNCTION DOC] HELPER: Compile start callback printing a message; ignores its
            # argument.
            print("Compilation started.")

        @torch._dynamo.on_compile_end
        def end_callback(_):
            # [FUNCTION DOC] HELPER: Compile end callback printing a message; ignores its
            # argument.
            print("Compilation ended.")

        mod = ToyModel()
        x = torch.randn(10, 10)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_mod = torch.compile(backend="eager", fullgraph=True)(mod)
            opt_mod(x)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(printed_output, "Compilation started.\nCompilation ended.")

    def test_compile_eager_options(self):
        # [FUNCTION DOC] Smoke test: compile with the eager and aot_eager backends passing an
        # unrecognized option mapping and call each once; passes if no exception is raised.
        @torch.compile(backend="eager", options={"foo": 2})
        def f(x):
            # [FUNCTION DOC] HELPER: Return x plus x.
            return x + x

        f(torch.randn(3))

        @torch.compile(backend="aot_eager", options={"foo": 2})
        def g(x):
            # [FUNCTION DOC] HELPER: Return x plus x.
            return x + x

        g(torch.randn(3))

    def test_compilation_callback_with_graph_break(self):
        # [FUNCTION DOC] Register start/end callbacks that increment a shared counter and print
        # it, compile a function with a graph break and run it with stdout captured; assert the
        # printed counters are 1 through 4 (a start and end for each of the two compiled
        # frames).
        torch._dynamo.reset()
        counter = 0

        @torch._dynamo.on_compile_start
        def start_callback(_):
            # [FUNCTION DOC] HELPER: Increment the shared counter and print its value.
            nonlocal counter
            counter += 1
            print(f"Counter = {counter}")

        @torch._dynamo.on_compile_end
        def end_callback(_):
            # [FUNCTION DOC] HELPER: Increment the shared counter and print its value.
            nonlocal counter
            counter += 1
            print(f"Counter = {counter}")

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Add 1, force a graph break, return sin of the result.
            x = x + 1
            torch._dynamo.graph_break()
            return torch.sin(x)

        x = torch.randn(10, 10)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            fn(x)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(
            printed_output, "Counter = 1\nCounter = 2\nCounter = 3\nCounter = 4"
        )

    def test_compilation_constant_hasattr_fail(self):
        # [FUNCTION DOC] Assert that calling a compiled function that invokes a tensor method on
        # None raises AttributeError (falls back to eager rather than an internal Dynamo error).
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Return x.max().
            return x.max()

        # We should fallback to normal mode, and throw a AttributeError, not a internal dynamo exception
        with self.assertRaises(AttributeError):
            fn(None)

    def test_compilation_evnum_hasattr_fail(self):
        # [FUNCTION DOC] Assert that calling a compiled function that invokes a tensor method on
        # an Enum member raises AttributeError, not an internal Dynamo error.
        from enum import Enum

        class TestEnum(Enum):
            # [CLASS DOC] HELPER: Enum with a single member VALID.
            VALID = 1

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Return x.max().
            return x.max()

        # We should fallback to normal mode, and throw a AttributeError, not a internal dynamo exception
        with self.assertRaises(AttributeError):
            fn(TestEnum.VALID)

    def test_compilation_name_error(self):
        # [FUNCTION DOC] Assert a compiled function that calls an undefined name raises
        # NameError.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Add 1 then call an undefined function.
            x = x + 1
            does_not_exist()  # noqa: F821
            return x

        x = torch.randn(10, 10)
        with self.assertRaises(NameError):
            fn(x)

    def test_compilation_tensor_invalid_method(self):
        # [FUNCTION DOC] Assert a compiled function calling a nonexistent tensor method raises
        # AttributeError.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Build a tensor from x and call a nonexistent method on it.
            y = torch.tensor(x)
            return y.doesnotexist()

        x = torch.randn(10, 10)

        with self.assertRaises(AttributeError):
            fn(x)

    def test_torch_script_compilation(self):
        # [FUNCTION DOC] Compile a torch.jit.script function with the eager backend, call it and
        # assert the output equals the input.
        @torch.jit.script
        def fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Scripted identity function.
            return x

        a = torch.randn(1, 1)
        out = torch.compile(fn, backend="eager")(a)
        self.assertEqual(out, a)

    def test_compile_script_module_error(self):
        # [FUNCTION DOC] Assert that compiling a scripted module raises RuntimeError saying
        # torch.compile does not support torch.jit.script.
        model = torch.nn.Sequential(torch.nn.Linear(3, 3))
        model.eval()
        scripted = torch.jit.script(model)
        with self.assertRaisesRegex(
            RuntimeError, "torch.compile does not support compiling torch.jit.script"
        ):
            torch.compile(scripted, backend="eager")

    def test_compile_frozen_module_error(self):
        # [FUNCTION DOC] Script and freeze a module (expecting a FutureWarning) and assert
        # compiling it with the eager backend raises RuntimeError that compiling jit.script is
        # unsupported.
        model = torch.nn.Sequential(torch.nn.Linear(3, 3))
        model.eval()
        scripted = torch.jit.script(model)
        with self.assertWarns(FutureWarning):
            frozen = torch.jit.freeze(scripted)
        with self.assertRaisesRegex(
            RuntimeError, "torch.compile does not support compiling torch.jit.script"
        ):
            torch.compile(frozen, backend="eager")

    def test_compile_frozen_module_inductor_error(self):
        # [FUNCTION DOC] Same as the frozen module test but with the inductor backend: assert
        # RuntimeError about torch.jit.script being unsupported.
        model = torch.nn.Sequential(torch.nn.Linear(3, 3))
        model.eval()
        scripted = torch.jit.script(model)
        with self.assertWarns(FutureWarning):
            frozen = torch.jit.freeze(scripted)
        with self.assertRaisesRegex(
            RuntimeError, "torch.compile does not support compiling torch.jit.script"
        ):
            torch.compile(frozen, backend="inductor")

    def test_inline_script_module_graph_break(self):
        # [FUNCTION DOC] Compile a module containing a scripted submodule with a counting
        # backend and assert zero frames compiled (the ScriptModule causes a graph break).
        class OuterModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper module that stores a submodule and applies it in
            # forward.
            def __init__(self, sub):
                # [FUNCTION DOC] HELPER: Store the submodule as sub.
                super().__init__()
                self.sub = sub

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return sub applied to x.
                return self.sub(x)

        inner = torch.nn.Linear(3, 3)
        scripted_inner = torch.jit.script(inner)
        outer = OuterModule(scripted_inner)

        cnt = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(outer, backend=cnt)
        inputs = torch.randn(2, 3)
        compiled(inputs)
        # ScriptModule submodule causes a graph break
        self.assertEqual(cnt.frame_count, 0)

    def test_to_sparse_to_dense_with_graph_break(self):
        # [FUNCTION DOC] Compile with the eager backend a function converting a tensor to sparse
        # and back to dense, and assert the output equals the eager result.
        def fn(x):
            # [FUNCTION DOC] HELPER: Convert x to sparse then to dense and return it.
            x = x.to_sparse()
            x = x.to_dense()
            return x

        x = torch.tensor([[1.0]])
        c_fn = torch.compile(fn, backend="eager")

        output = fn(x)
        c_output = c_fn(x)
        self.assertEqual(output, c_output)

    def test_list_bad_access(self):
        # [FUNCTION DOC] Assert a compiled function indexing a list with an out-of-range runtime
        # integer raises IndexError.
        @torch.compile(backend="eager")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Put x in a list and return the element at index y.
            a = [x]
            return a[y]

        with self.assertRaises(IndexError):
            fn(torch.randn(10), 99)


# The private variants of the below functions are extensively tested
# So as long as the signatures match we're good
class PublicTorchCompilerTests(TestCase):
# [CLASS DOC] Test case checking that public torch.compiler functions have signatures compatible
# with their torch._dynamo counterparts.
    def check_signature(self, public_fn_name, private_fn_name, private_namespace):
        # [FUNCTION DOC] HELPER: Look up the public function in torch.compiler and the private
        # one in the given namespace, and assert that their signatures are equal, or that the
        # public one has fewer parameters and each of its parameters equals the corresponding
        # private parameter in order. Fails with a message showing both signatures.
        public_fn = getattr(torch.compiler, public_fn_name)
        private_fn = getattr(private_namespace, private_fn_name)

        public_sig = inspect.signature(public_fn)
        private_sig = inspect.signature(private_fn)

        matching = public_sig == private_sig
        matching |= len(public_sig.parameters) < len(private_sig.parameters) and all(
            public == private
            for public, private in zip(
                public_sig.parameters.items(), private_sig.parameters.items()
            )
        )

        self.assertEqual(
            matching,
            True,
            lambda msg: f"{msg}\nSignatures do not match for function {public_fn_name}() \n Public: {public_sig} \n Private: {private_sig}",
        )

    def test_dynamo_signatures(self):
        # [FUNCTION DOC] Run the signature check for reset, allow_in_graph, list_backends,
        # assume_constant_result and disable against torch._dynamo.
        function_names = [
            "reset",
            "allow_in_graph",
            "list_backends",
            "assume_constant_result",
            "disable",
        ]

        for fn_name in function_names:
            self.check_signature(fn_name, fn_name, torch._dynamo)


class FullgraphTests(TestCase):
# [CLASS DOC] Test case for fullgraph mode errors when Dynamo skips a frame.
    def test_fullgraph_errors_on_frame_skip_with_dispatch_mode(self):
        # [FUNCTION DOC] With a non-infra TorchDispatchMode active, compiling a function with
        # fullgraph raises RuntimeError mentioning the dispatch mode.
        from torch.utils._python_dispatch import TorchDispatchMode

        class SkipMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that forwards each call to the original
            # function.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Call the intercepted func with the given args and
                # kwargs and return its result.
                return func(*args, **kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Return x plus 1.
            return x + 1

        x = torch.randn(5)
        with SkipMode():
            with self.assertRaisesRegex(
                RuntimeError, "non-infra torch dispatch mode present"
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(x)

    def test_fullgraph_errors_on_frame_skip_dynamo_disabled(self):
        # [FUNCTION DOC] With dynamo config disable set, compiling and calling a fullgraph
        # function raises RuntimeError saying Dynamo tracing is disabled.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return x plus 1.
            return x + 1

        x = torch.randn(5)
        with torch._dynamo.config.patch(disable=True):
            with self.assertRaisesRegex(RuntimeError, "Dynamo tracing is disabled"):
                torch.compile(fn, backend="eager", fullgraph=True)(x)

    def test_fullgraph_empty_graph_no_error(self):
        # [FUNCTION DOC] A fullgraph function that traces no tensor ops (returns a length) must
        # not raise and returns 5.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return len(x).
            return len(x)

        x = torch.randn(5)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, 5)

    def test_fullgraph_skip_reason_message(self):
        # [FUNCTION DOC] With dynamo disabled, fullgraph compile raises RuntimeError whose
        # munged message lists the skipped frame my_function with reason Dynamo tracing is
        # disabled (compared to an inline expected string); fails the test if nothing is raised.
        def my_function(x):
            # [FUNCTION DOC] HELPER: Return x plus 1.
            return x + 1

        x = torch.randn(5)
        with torch._dynamo.config.patch(disable=True):
            try:
                torch.compile(my_function, backend="eager", fullgraph=True)(x)
                self.fail("Expected RuntimeError")
            except RuntimeError as e:
                self.assertExpectedInline(
                    munge_exc(str(e)),
                    """\
torch.compile with fullgraph=True found no compiled frames. Skipped frames:
  - my_function (test_compile.py:N): Dynamo tracing is disabled""",
                )

    def test_fullgraph_no_error_with_non_default_stance(self):
        # [FUNCTION DOC] Compile a fullgraph function with dynamic False, call it, then call
        # with a different integer under the eager_on_recompile stance; assert no error and the
        # correct result.
        @torch.compile(fullgraph=True, backend="eager", dynamic=False)
        def fn(x, n):
            # [FUNCTION DOC] HELPER: Return x plus n.
            return x + n

        fn(torch.ones(3), 1)
        with torch.compiler.set_stance("eager_on_recompile"):
            result = fn(torch.ones(3), 2)
        self.assertEqual(result, torch.ones(3) + 2)

    def test_fullgraph_exported_module_no_error(self):
        # [FUNCTION DOC] Export a simple module with torch.export and call the exported module's
        # module(); assert the result equals x plus 1 with no error.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds 1.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x plus 1.
                return x + 1

        m = M()
        x = torch.randn(5)
        exported = torch.export.export(m, (x,))
        result = exported.module()(x)
        self.assertEqual(result, x + 1)


if __name__ == "__main__":
    run_tests()
