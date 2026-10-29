# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the post-AOT repro tooling in torch._dynamo.repro.after_aot: saving
# Inductor args (including symbolic ints), generating standalone repro scripts with
# save_graph_repro, serializing inputs with InputWriter/InputReader, choosing compile args for
# symbolic versus real tracing, and accelerator synchronization in the debug wrappers.
# Importing the module has no side effects beyond imports; device-generic tests are
# instantiated at import time for TestWrapCompilerDebugSync.

import importlib
import io
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch._dynamo.test_case
from torch._dynamo.repro.after_aot import (
    _extract_distributed_info,
    _get_compile_args,
    InputReader,
    InputWriter,
    repro_minify,
    repro_run,
    save_graph_repro,
)
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import IS_FBCODE, TEST_CUDA
from torch.utils._traceback import report_compile_source_on_error
from torch.utils._triton import has_triton


def strip_trailing_whitespace(r):
    # [FUNCTION DOC] HELPER: Returns the input string with trailing whitespace removed from every
    # line, keeping line breaks.
    return "\n".join([l.rstrip() for l in r.split("\n")])


class TestAfterAot(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for torch._dynamo.repro.after_aot repro generation and input
    # serialization. Stateless; each test builds its own graphs, temporary directories and mocks.
    @patch("torch.cuda.is_available", lambda: False)
    def test_save_args_dynamic_shapes(self):
        # [FUNCTION DOC] With CUDA reported unavailable, compiles a function dynamically with
        # the save_args option and caches disabled, redirecting the temp dir used by the
        # inductor debug module to a temporary directory. Asserts the compiled result matches
        # eager, that a saved-args directory with at least one file appears, and that
        # reloading the first file via load_args_and_run_compile_fx_inner returns a non-None
        # result.
        import torch._functorch.config as functorch_config
        from torch._inductor import config as inductor_config
        from torch._inductor.debug import load_args_and_run_compile_fx_inner

        torch._dynamo.reset()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a zeros tensor shaped like the input after
            # reading its element count (to introduce a dynamic-shape use).
            output = torch.zeros_like(x)
            _ = output.numel()
            return output

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("torch._inductor.debug.tempfile.gettempdir", return_value=tmpdir),
            inductor_config.patch(
                {
                    "save_args": True,
                    "force_disable_caches": True,
                    "fx_graph_cache": False,
                    "fx_graph_remote_cache": False,
                }
            ),
            functorch_config.patch({"enable_autograd_cache": False}),
        ):
            opt_fn = torch.compile(fn, dynamic=True)  # noqa: UNSPECIFIED_BACKEND
            inp = torch.randn(4)
            self.assertEqual(opt_fn(inp), fn(inp))

            saved_args_dir = os.path.join(tmpdir, "inductor_saved_args")
            self.assertTrue(os.path.isdir(saved_args_dir))
            saved_args_paths = os.listdir(saved_args_dir)
            self.assertGreater(len(saved_args_paths), 0)

            compiled = load_args_and_run_compile_fx_inner(
                os.path.join(saved_args_dir, saved_args_paths[0])
            )
            self.assertIsNotNone(compiled)

    def test_save_args_custom_sympy_expr(self):
        # [FUNCTION DOC] Creates a symbolic int whose expression is a modular indexing of a
        # dynamic symbol, saves it with save_args_for_compile_fx_inner into a temporary
        # directory, and unpickles the file to check the stored argument is a
        # SymExprMetadataHolder holding that expression. Then reloads it with the compile
        # entry point patched by a fake, asserting the restored value is a SymInt with the
        # same expression and hint 0.
        from torch._dynamo.source import ConstantSource
        from torch._inductor.debug import (
            load_args_and_run_compile_fx_inner,
            save_args_for_compile_fx_inner,
            SymExprMetadataHolder,
        )
        from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
        from torch.utils._sympy.functions import ModularIndexing

        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(
            17,
            ConstantSource("custom_symbol"),
            dynamic_dim=DimDynamic.DYNAMIC,
            do_not_specialize_zero_one=True,
        )
        expr = ModularIndexing(symbol + 3, 2, 5)
        symint = shape_env.create_symintnode(expr, hint=0)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("torch._inductor.debug.tempfile.gettempdir", return_value=tmpdir),
        ):
            save_args_for_compile_fx_inner(symint)
            saved_args_dir = os.path.join(tmpdir, "inductor_saved_args")
            saved_path = os.path.join(saved_args_dir, os.listdir(saved_args_dir)[0])

            with open(saved_path, "rb") as f:
                saved_args, _ = pickle.load(f)
            self.assertIsInstance(saved_args[0], SymExprMetadataHolder)
            self.assertTrue(saved_args[0].expr.has(ModularIndexing))

            def fake_compile_fx_inner(restored_symint):
                # [FUNCTION DOC] HELPER: Fake compile_fx_inner that asserts its
                # argument is a SymInt with the modular indexing expression and hint
                # 0, then returns it.
                self.assertIsInstance(restored_symint, torch.SymInt)
                self.assertTrue(restored_symint.node.expr.has(ModularIndexing))
                self.assertEqual(restored_symint.node.hint, 0)
                return restored_symint

            with patch(
                "torch._inductor.compile_fx.compile_fx_inner", fake_compile_fx_inner
            ):
                restored = load_args_and_run_compile_fx_inner(saved_path)

            self.assertIsInstance(restored, torch.SymInt)
            self.assertTrue(restored.node.expr.has(ModularIndexing))

    def test_repro_minify_disables_repro_after(self):
        # [FUNCTION DOC] Checks repro_minify temporarily disables the repro_after dynamo
        # config while minifying. With repro_after set to aot and repro_common and the
        # minifier patched by fakes, runs repro_minify; the fake minifier records the config
        # value it sees, and the test asserts it was None.
        seen_repro_after = object()

        def fake_repro_common(options, mod, load_args):
            # [FUNCTION DOC] HELPER: Fake repro_common returning a traced graph (identity
            # returning a tuple) and its single random input.
            def f(x):
                return (x,)

            args = [torch.randn(1)]
            return make_fx(f)(*args), args

        def fake_minifier(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Fake minifier that stores the current repro_after
            # config value in an enclosing variable.
            nonlocal seen_repro_after
            seen_repro_after = torch._dynamo.config.repro_after

        options = SimpleNamespace(
            accuracy="accuracy",
            check_str=None,
            isolate=False,
            save_dir=None,
            tracing_mode="real",
            offload_to_disk=False,
            skip_saving_eager_intermediates=False,
            skip_sanity=False,
            max_granularity=None,
        )

        with (
            torch._dynamo.config.patch(repro_after="aot"),
            patch("torch._dynamo.repro.after_aot.repro_common", fake_repro_common),
            patch("functorch.compile.minifier", fake_minifier),
        ):
            repro_minify(options, torch.nn.Identity(), lambda reader: None)

        self.assertIsNone(seen_repro_after)

    @unittest.skipIf(IS_FBCODE, "NotImplementedError")
    def test_save_graph_repro(self):
        # [FUNCTION DOC] Generates a repro script with save_graph_repro for a simple traced
        # multiply graph, executing it twice via exec: once with saved storages in a temporary
        # save dir, and once after deleting the storages directory (it must still run). If
        # Triton is available, first snapshots and resets the kernel side table and registers
        # a small copy kernel, restoring the snapshot at the end. Skipped in fbcode.
        # KNOWN ISSUE: the kernel side table is restored only on success, not in a finally
        #   block, so a failure leaves it modified. The TODO notes CUDA context initialization
        #   even on CPU.
        # TODO: This triggers CUDA context initialization, even though
        # it is CPU only
        saved_kernel_state = None
        if has_triton():
            import triton
            import triton.language as tl

            saved_kernel_state = (
                dict(kernel_side_table.id_to_kernel),
                dict(kernel_side_table.kernel_to_id),
                dict(kernel_side_table.constant_args),
            )
            kernel_side_table.reset_table()

            # [FUNCTION DOC] HELPER: Triton kernel copying a masked block of input
            # to output; registered in the kernel side table so the generated
            # repro must serialize it.
            @triton.jit
            def _repro_kernel(x_ptr, y_ptr, size, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offsets < size
                tl.store(
                    y_ptr + offsets,
                    tl.load(x_ptr + offsets, mask=mask),
                    mask=mask,
                )

            kernel_side_table.add_kernel(_repro_kernel)

        buf = io.StringIO()
        args = [torch.randn(4)]

        def f(x):
            # [FUNCTION DOC] HELPER: Traced function returning a one-element tuple with
            # the squared input.
            return (x * x,)

        gm = make_fx(f)(*args)
        with tempfile.TemporaryDirectory() as d:
            save_graph_repro(buf, gm, args, "inductor_accuracy", save_dir=d)
            r = buf.getvalue()
            with report_compile_source_on_error():
                exec(r, {"__compile_source__": r})

            shutil.rmtree(os.path.join(d, "storages"))

            # Should still work even without the save dir
            with report_compile_source_on_error():
                exec(r, {"__compile_source__": r})

        if saved_kernel_state is not None:
            (
                kernel_side_table.id_to_kernel,
                kernel_side_table.kernel_to_id,
                kernel_side_table.constant_args,
            ) = saved_kernel_state

    @unittest.skipIf(sys.byteorder != "little", "checksum depends on endianness")
    def test_dump_tensor(self):
        # [FUNCTION DOC] Checks InputWriter output for tensors. For each of three tensors
        # (contiguous float, int32 with dtype hint, channels-last with explicit strides),
        # writes it with stable hashing, compares the generated lines to expected text
        # (storage hash and size, tensor call with shape, dtype, strides), then executes the
        # lines with an InputReader and asserts the reconstructed argument equals the
        # original. Skipped on big-endian since the checksum depends on it.
        def test(tensor, expected):
            # [FUNCTION DOC] HELPER: Runs one write/compare/execute round trip in a fresh
            # temporary directory for the given tensor and expected text.
            with tempfile.TemporaryDirectory() as d:
                writer = InputWriter(d, stable_hash=True)
                writer.tensor("x", tensor)
                self.assertExpectedInline("\n".join(writer._lines), expected, skip=1)
                reader = InputReader(d)
                env = {"reader": reader, "torch": torch}
                # TODO: assert no logs
                exec("\n".join(writer._lines), env)
                self.assertEqual(reader.args[0], tensor)

        test(
            torch.zeros(3, 4),
            """\
buf0 = reader.storage('c17fd92682ca5b304ac71074b558dda9e8eb4d66', 48)
reader.tensor(buf0, (3, 4), is_leaf=True)  # x""",
        )
        test(
            torch.ones(3, 4, dtype=torch.int32),
            """\
buf0 = reader.storage('7c221e2da0c58c700cc2996644dd13d042bd552e', 48, dtype_hint=torch.int32)
reader.tensor(buf0, (3, 4), dtype=torch.int32, is_leaf=True)  # x""",
        )
        test(
            torch.empty((3, 4, 5, 6), memory_format=torch.channels_last).fill_(2),
            """\
buf0 = reader.storage('49ebab3961d6221e64c4c72b0aefd976bdd2afc4', 1440)
reader.tensor(buf0, (3, 4, 5, 6), (120, 1, 24, 4), is_leaf=True)  # x""",
        )

    def test_dump_opaque(self):
        # [FUNCTION DOC] Builds a graph with an extra placeholder and passes a
        # FakeScriptObject (class name MyClass) as an argument to save_graph_repro; asserts
        # the generated script contains an opaque reader call with that class name.
        """save_graph_repro should emit reader.opaque() for FakeScriptObject args."""
        from torch._library.fake_class_registry import FakeScriptObject

        fake_obj = FakeScriptObject(object(), "__torch__.MyClass", None)

        def f(x):
            # [FUNCTION DOC] HELPER: Traced function returning a one-element tuple with
            # the squared input.
            return (x * x,)

        args = [torch.randn(4), fake_obj]
        gm = make_fx(f)(args[0])
        with gm.graph.inserting_before(next(iter(gm.graph.nodes))):
            gm.graph.placeholder("obj")
        gm.recompile()

        buf = io.StringIO()
        save_graph_repro(buf, gm, args, "inductor_accuracy")
        r = buf.getvalue()
        self.assertIn("reader.opaque('__torch__.MyClass')", r)

    def test_save_graph_repro_emits_custom_backend_codegen_config(self):
        # [FUNCTION DOC] Loads the custom_inductor_config test module from the inductor test
        # directory (temporarily on sys.path), registers the CPU backend with it as a custom
        # backend config and enables its enable_optimisation option, then generates an
        # inductor repro with save_graph_repro. Asserts the script imports that config module
        # and sets the option to True.
        from torch.testing._internal.inductor_utils import patch_inductor_backend

        # Reuse the custom ConfigModule already used by Inductor cache-key tests
        # instead of creating a second dummy config module for repro coverage.
        test_inductor_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "inductor")
        )
        with patch.object(sys, "path", [test_inductor_dir, *sys.path]):
            custom_inductor_config = importlib.import_module("custom_inductor_config")

        # This re-registers the CPU backend with its original scheduling and
        # wrapper codegen, but adds a third-party backend config module.
        with (
            patch_inductor_backend("cpu", custom_backend_config=custom_inductor_config),
            custom_inductor_config.patch(enable_optimisation=True),
        ):
            args = [torch.randn(4)]

            def f(x):
                # [FUNCTION DOC] HELPER: Traced function returning a one-element tuple
                # with the input plus one.
                return (x + 1,)

            gm = make_fx(f)(*args)
            buf = io.StringIO()
            save_graph_repro(buf, gm, args, "inductor")
            repro = buf.getvalue()

            self.assertIn(f"import {custom_inductor_config.__name__}", repro)
            self.assertIn(
                f"{custom_inductor_config.__name__}.enable_optimisation = True",
                repro,
            )

    def import_triton_extra_import_kernel(self):
        # [FUNCTION DOC] HELPER: Imports and returns the _triton_extra_import_kernel module
        # from the dynamo test directory, temporarily placing that directory first on
        # sys.path. Not itself a test (name lacks the test prefix).
        test_dynamo_dir = os.path.abspath(os.path.dirname(__file__))
        with patch.object(sys, "path", [test_dynamo_dir, *sys.path]):
            return importlib.import_module("_triton_extra_import_kernel")

    @unittest.skipIf(not has_triton(), "requires Triton")
    def test_save_graph_repro_emits_extra_triton_imports(self):
        # [FUNCTION DOC] Registers a kernel that uses a libdevice import in a patched, empty
        # kernel side table, generates an inductor repro with save_graph_repro for a simple
        # graph, and asserts the script contains the libdevice import line. Skipped without
        # Triton.
        extra_import_kernel = self.import_triton_extra_import_kernel()

        with (
            patch.object(kernel_side_table, "id_to_kernel", {}),
            patch.object(kernel_side_table, "kernel_to_id", {}),
            patch.object(kernel_side_table, "constant_args", {}),
        ):
            # The helper kernel references `libdevice` directly instead of only
            # `triton` or `tl`, matching user kernels that import helpers from
            # Triton's extra packages.
            kernel_side_table.add_kernel(
                extra_import_kernel.triton_kernel_with_extra_import
            )

            args = [torch.randn(4)]

            def f(x):
                # [FUNCTION DOC] HELPER: Traced function returning a one-element tuple
                # with the input plus one.
                return (x + 1,)

            gm = make_fx(f)(*args)
            buf = io.StringIO()
            save_graph_repro(buf, gm, args, "inductor")
            repro = buf.getvalue()

            self.assertIn("import triton.language.extra.libdevice as libdevice", repro)

    @unittest.skipIf(not TEST_CUDA or not has_triton(), "requires CUDA and Triton")
    def test_after_aot_minifier_emits_extra_triton_imports_integration(self):
        # [FUNCTION DOC] Integration test: compiles a function launching the extra-import
        # Triton kernel with repro_after aot, repro level 2 and a temporary debug dir, while
        # SIMDScheduling.codegen_node is patched to raise. Asserts the call raises
        # RuntimeError, then reads the generated minifier_launcher.py and asserts it has the
        # libdevice import. Needs CUDA and Triton.
        extra_import_kernel = self.import_triton_extra_import_kernel()
        from torch._dynamo.debug_utils import minifier_dir
        from torch._inductor.codegen.simd import SIMDScheduling

        @torch.compile  # noqa: UNSPECIFIED_BACKEND
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function that launches the Triton kernel
            # then returns the input plus one.
            extra_import_kernel.triton_kernel_with_extra_import[(1,)](x, BLOCK=16)
            return x + 1

        def fail_codegen_node(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement for codegen_node that always raises
            # RuntimeError to force the minifier path.
            raise RuntimeError("intentional triton codegen failure")

        with tempfile.TemporaryDirectory() as d:
            repro_path = None
            with (
                torch._dynamo.config.patch(
                    repro_after="aot",
                    repro_level=2,
                    debug_dir_root=d,
                ),
                patch.object(
                    SIMDScheduling,
                    "codegen_node",
                    side_effect=fail_codegen_node,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    fn(torch.randn(16, device="cuda"))
                repro_path = os.path.join(minifier_dir(), "minifier_launcher.py")

            self.assertIsNotNone(repro_path)
            with open(repro_path) as f:
                repro = f.read()

        self.assertIn("import triton.language.extra.libdevice as libdevice", repro)

    def test_repro_run_accuracy_does_not_reuse_compiled_graph(self):
        # [FUNCTION DOC] Runs repro_run in accuracy mode on a module doing a foreach add and
        # in-place copies on three scalar tensors, with a loader that sets three random float
        # scalars as args. Passes if no error is raised, guarding against reusing a previously
        # compiled graph in the accuracy check.
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three inputs that adds one to each via
            # foreach, copies results back in place and returns an empty tuple.
            def forward(self, arg0, arg1, arg2):
                # [FUNCTION DOC] HELPER: Applies foreach add of one to the three
                # inputs, copies each result into its input, returns an empty tuple.
                vals = torch.ops.aten._foreach_add.Scalar([arg0, arg1, arg2], 1)
                torch.ops.aten.copy_.default(arg0, vals[0])
                torch.ops.aten.copy_.default(arg1, vals[1])
                torch.ops.aten.copy_.default(arg2, vals[2])
                return ()

        def load_args(reader):
            # [FUNCTION DOC] HELPER: Sets the reader args to three random float32 scalar
            # tensors.
            reader.args = [torch.rand((), dtype=torch.float32) for _ in range(3)]

        load_args._version = 0
        options = SimpleNamespace(
            accuracy="accuracy", save_dir=None, tracing_mode="real"
        )
        repro_run(options, Repro(), load_args)

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_dump_generator(self):
        # [FUNCTION DOC] Clones the state of CUDA generator 0, writes it with InputWriter
        # under the name fwd_rng_state_0, checks the single generated line, then executes it
        # with an InputReader and asserts the arg is a Generator on cuda device 0. Needs CUDA.
        gen = torch.cuda.default_generators[0].clone_state()
        writer = InputWriter(None)
        writer.generator("fwd_rng_state_0", gen)
        self.assertExpectedInline(
            "\n".join(writer._lines),
            """reader.generator('cuda', 0)  # fwd_rng_state_0""",
        )
        reader = InputReader(None)
        env = {"reader": reader, "torch": torch}
        exec("\n".join(writer._lines), env)
        self.assertIsInstance(reader.args[0], torch._C.Generator)
        self.assertEqual(reader.args[0].device, torch.device("cuda", 0))

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_graphsafe_rng_repro(self):
        # [FUNCTION DOC] Traces a graph, adds a rng-state placeholder, and calls
        # save_graph_repro with a CUDA tensor and a cloned CUDA generator. Asserts the script
        # contains a generator reader call for cuda 0 and no unsupported marker. Needs CUDA.
        """save_graph_repro should emit reader.generator() for Generator args."""
        gen = torch.cuda.default_generators[0].clone_state()

        def f(x):
            return (x * x,)

        args = [torch.randn(4, device="cuda"), gen]
        gm = make_fx(f)(args[0])
        with gm.graph.inserting_before(next(iter(gm.graph.nodes))):
            gm.graph.placeholder("fwd_rng_state_0")
        gm.recompile()

        buf = io.StringIO()
        save_graph_repro(buf, gm, args, "inductor_accuracy")
        r = buf.getvalue()
        self.assertIn("reader.generator('cuda', 0)", r)
        self.assertNotIn("reader.unsupported(", r)

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_extract_distributed_info_skips_non_string_group_name(self):
        # [FUNCTION DOC] Builds a graph with an all_reduce collective whose group_name is a
        # graph placeholder (a Node, not a string) and asserts _extract_distributed_info
        # returns an empty mapping. Skipped when distributed is unavailable.
        """_extract_distributed_info should skip ops where group_name is an FX Node."""
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        g = gm.graph
        x = g.placeholder("x")
        group_name = g.placeholder("group_name")
        ar = g.call_function(
            torch.ops._c10d_functional.all_reduce.default,
            args=(x, "sum", group_name),
        )
        g.output(ar)
        gm.recompile()

        result = _extract_distributed_info(gm)
        self.assertEqual(result, {})

    def test_get_compile_args_symbolic_tracing(self):
        # [FUNCTION DOC] Traces with symbolic mode with a tensor and an int, and asserts
        # _get_compile_args returns a new list whose first item is a FakeTensor and second a
        # SymInt.
        """_get_compile_args extracts FakeTensor/SymInt from symbolic tracing."""

        def f(x, size):
            # [FUNCTION DOC] HELPER: Traced function returning a one-element tuple with
            # the input sliced to the given size.
            return (x[:size],)

        args = [torch.randn(10), 5]
        gm = make_fx(f, tracing_mode="symbolic")(*args)
        result = _get_compile_args(gm, args)
        # Should NOT be the same object — metadata was extracted
        self.assertIsNot(result, args)
        # First element should be a FakeTensor (not a concrete tensor)
        self.assertIsInstance(result[0], torch._subclasses.FakeTensor)
        # Second element should be a SymInt (not a concrete int)
        self.assertIsInstance(result[1], torch.SymInt)

    def test_get_compile_args_preserves_shapes(self):
        # [FUNCTION DOC] Traces a multiply symbolically with a 4 by 8 tensor and asserts the
        # FakeTensor from _get_compile_args keeps shape 4 by 8.
        """_get_compile_args preserves symbolic shape info from traced graph."""

        def f(x):
            # [FUNCTION DOC] HELPER: Traced function returning a one-element tuple with
            # the input doubled.
            return (x * 2,)

        args = [torch.randn(4, 8)]
        gm = make_fx(f, tracing_mode="symbolic")(*args)
        result = _get_compile_args(gm, args)
        # FakeTensor should preserve the shape
        self.assertEqual(result[0].shape, torch.Size([4, 8]))

    def test_get_compile_args_real_tracing_returns_concrete(self):
        # [FUNCTION DOC] Traces in real mode and asserts _get_compile_args returns the very
        # same args object, since there are no SymInt placeholders (avoids FakeTensorMode
        # mismatch).
        """_get_compile_args returns original args for real-mode tracing.

        make_fx with tracing_mode='real' produces FakeTensors in placeholder
        metadata, but from different FakeTensorModes.  Extracting these would
        cause a FakeTensorMode mismatch in Inductor.  _get_compile_args must
        detect this and return concrete args instead.
        """

        def f(x, y):
            # [FUNCTION DOC] HELPER: Traced function returning a one-element tuple with
            # the sum of two inputs.
            return (x + y,)

        args = [torch.randn(4), torch.randn(4)]
        gm = make_fx(f, tracing_mode="real")(*args)
        result = _get_compile_args(gm, args)
        # For real tracing (no SymInt inputs), should return args as-is
        self.assertIs(result, args)

    def test_get_compile_args_empty_graph(self):
        # [FUNCTION DOC] Builds an empty graph with no placeholders and asserts
        # _get_compile_args returns the same args object.
        """_get_compile_args handles empty graph with no placeholders."""
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        gm.graph.output(None)
        gm.recompile()
        args = [torch.randn(4)]
        result = _get_compile_args(gm, args)
        self.assertIs(result, args)

    def test_get_compile_args_e2e_symbolic_compile(self):
        # [FUNCTION DOC] End-to-end: traces a slice-and-scale function symbolically, compiles
        # with compile_fx_inner using concrete args and then using args from
        # _get_compile_args, runs the second compiled result, and asserts the output has shape
        # (16,). Neither compile result may be a string.
        """E2E: _get_compile_args produces args usable by compile_fx_inner
        for symbolically-traced graphs (regression test for the
        'NameError: name s48 is not defined' issue in fx_graph_runnable
        repro scripts).
        """
        from torch._inductor.compile_fx import compile_fx_inner

        def f(n, x, y):
            # [FUNCTION DOC] HELPER: Traced function slicing two tensors to a length n,
            # scaling one by n minus one, and returning their sum in a tuple.
            sliced = x[:n]
            scaled = sliced * (n - 1)
            return (scaled + y[:n],)

        N = 16
        concrete_args = [N, torch.randn(32), torch.randn(32)]
        gm = make_fx(f, tracing_mode="symbolic")(*concrete_args)

        compiled = compile_fx_inner(gm, concrete_args)
        self.assertNotIsInstance(compiled, str)
        compiled(list(concrete_args))

        symbolic_args = _get_compile_args(gm, concrete_args)
        compiled = compile_fx_inner(gm, symbolic_args)
        self.assertNotIsInstance(compiled, str)
        result = compiled(list(concrete_args))
        self.assertEqual(result[0].shape, torch.Size([N]))

    def test_get_compile_args_e2e_real_no_fake_mode_mismatch(self):
        # [FUNCTION DOC] End-to-end: traces in real mode, confirms the placeholder FakeTensors
        # come from more than one fake mode, asserts compile_fx_inner raises with a fake-mode
        # mismatch message when given those, then asserts _get_compile_args returns the
        # concrete args and compile_fx_inner on them works with output shape (4,).
        """E2E: compile_fx_inner fails when given FakeTensors from
        different FakeTensorModes (extracted from real-mode traced graph
        placeholder metadata) but succeeds with _get_compile_args which
        returns concrete args for real-mode tracing.

        This is the minimal repro for the FakeTensorMode mismatch
        AssertionError that affected 85/126 graphs in the model extractor.
        """
        from torch._inductor.compile_fx import compile_fx_inner

        def f(x, y):
            # [FUNCTION DOC] HELPER: Traced function returning a one-element tuple with
            # the sum of two inputs.
            return (x + y,)

        args = [torch.randn(4), torch.randn(4)]
        gm = make_fx(f, tracing_mode="real")(*args)

        # Verify that real-mode tracing creates FakeTensors with
        # different FakeTensorModes in placeholder metadata
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        fake_modes = set()
        for n in placeholders:
            val = n.meta.get("val")
            if isinstance(val, torch._subclasses.FakeTensor):
                fake_modes.add(id(val.fake_mode))
        self.assertGreater(len(fake_modes), 1, "Expected different FakeTensorModes")

        # BUG: manually extracting FakeTensors causes mode mismatch
        fake_args = [n.meta["val"] for n in placeholders]
        with self.assertRaisesRegex(Exception, "fake mode.*doesn't match"):
            compile_fx_inner(gm, fake_args)

        # FIX: _get_compile_args returns concrete args for real mode
        compile_args = _get_compile_args(gm, args)
        self.assertIs(compile_args, args)
        compiled = compile_fx_inner(gm, compile_args)
        self.assertNotIsInstance(compiled, str)
        result = compiled(list(args))
        self.assertEqual(result[0].shape, torch.Size([4]))

    def test_symint_expr_serialized(self):
        # [FUNCTION DOC] Creates symbolic ints from a shape env, writes a free symbol, a
        # derived symbol and a plain int with InputWriter, and asserts expressions appear for
        # the symbolic ones but not the plain int line. Executes the generated load_args with
        # an InputReader and asserts three args, a non-empty symint_exprs mapping and that
        # entry 1 contains a floor division.
        """InputWriter serializes symbolic expressions alongside hint values,
        and InputReader captures them in symint_exprs."""
        from torch._dynamo.source import ConstantSource
        from torch.fx.experimental.sym_node import SymNode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        s0 = shape_env.create_symbol(160, ConstantSource("s0"))
        s1 = shape_env.create_symbol(200, ConstantSource("s1"))

        si0 = torch.SymInt(SymNode(s0, shape_env, int, hint=160))
        si1 = torch.SymInt(SymNode(s1, shape_env, int, hint=200))
        derived = (si0 + si1) // 10

        writer = InputWriter(save_dir=None)
        writer.symint("free_sym", si0)
        writer.symint("derived_sym", derived)
        writer.symint("plain_int", 42)

        lines = writer.lines()
        code = "\n".join(lines)

        # Free symbol should have expr=
        self.assertIn("expr=", code)
        # Plain int should NOT have expr=
        plain_line = next(l for l in lines if "plain_int" in l)
        self.assertNotIn("expr=", plain_line)

        # Round-trip: execute load_args and verify reader captures exprs
        reader = InputReader(save_dir=None)
        ns: dict = {}
        exec(compile(code, "<test>", "exec"), ns)
        ns["load_args"](reader)

        self.assertEqual(len(reader.args), 3)
        self.assertTrue(hasattr(reader, "symint_exprs"))
        self.assertGreater(len(reader.symint_exprs), 0)
        # The derived symint should have an expression with '//'
        derived_expr = reader.symint_exprs.get(1)
        self.assertIsNotNone(derived_expr)
        self.assertIn("//", derived_expr)

    @unittest.skipIf(IS_FBCODE, "Subprocess spawning doesn't work in fbcode")
    def test_symint_expr_e2e_repro(self):
        # [FUNCTION DOC] Traces a graph symbolically with a SymInt and a tensor, generates an
        # inductor repro with save_graph_repro (symbolic tracing mode), asserts it serializes
        # the expression, writes it to a temporary file and runs it in a subprocess with a 60
        # second timeout, asserting exit code 0 (stderr tail shown on failure). Skipped in
        # fbcode.
        """End-to-end: generate a repro from a symbolically-traced graph
        with SymInt args, verify expr= appears, and run the repro."""
        import subprocess
        import tempfile

        from torch._dynamo.source import ConstantSource
        from torch.fx.experimental.sym_node import SymNode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        s0 = shape_env.create_symbol(8, ConstantSource("s0"))
        si0 = torch.SymInt(SymNode(s0, shape_env, int, hint=8))

        class SimpleGraph(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a tuple of the input's sum plus the
            # given size.
            def forward(self, size, x):
                # [FUNCTION DOC] HELPER: Returns a one-element tuple of the sum of x
                # plus size.
                return (x.sum() + size,)

        mod = SimpleGraph()
        args = [si0, torch.randn(16)]
        gm = make_fx(mod, tracing_mode="symbolic")(*args)

        buf = io.StringIO()
        save_graph_repro(
            buf,
            gm,
            args,
            "inductor",
            save_dir=None,
            tracing_mode="symbolic",
        )
        repro_src = buf.getvalue()

        # Verify expr= is serialized for the SymInt arg
        self.assertIn("expr=", repro_src)

        # Run the generated repro in a subprocess
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(repro_src)
            tmp.flush()
            res = subprocess.run(
                [sys.executable, tmp.name],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                res.returncode,
                0,
                lambda msg: f"{msg}\nRepro failed:\nSTDERR:\n{res.stderr[-1000:]}",
            )


class TupleOut(torch.nn.Module):
    # [CLASS DOC] HELPER: Module returning its input wrapped in a one-element tuple.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input in a tuple.
        return (x,)


class TestWrapCompilerDebugSync(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Device-generic tests that the repro debug helpers call the accelerator
    # synchronize function for non-CPU tensors. Instantiated per device by
    # instantiate_device_type_tests.
    def test_synchronize_called_for_accelerator_tensor(self, device):
        # [FUNCTION DOC] Skips on CPU. Wraps a fake compiler with wrap_compiler_debug under
        # repro_after aot, level 2, with accelerator synchronize patched to record calls,
        # compiles and runs it, and asserts synchronize was called.
        if torch.device(device).type == "cpu":
            self.skipTest("sync only triggered for non-CPU devices")

        from torch._dynamo.repro.after_aot import wrap_compiler_debug

        called = []

        def fake_compiler(gm, inputs, **kwargs):
            # [FUNCTION DOC] HELPER: Fake compiler returning a callable that adds one to
            # each tensor input.
            def compiled(real_inputs):
                # [FUNCTION DOC] HELPER: Returns the list of tensor inputs each
                # incremented by one.
                return [inp + 1 for inp in real_inputs if isinstance(inp, torch.Tensor)]

            return compiled

        wrapped = wrap_compiler_debug(fake_compiler, "test_backend")
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        inputs = [torch.randn(4, device=device)]

        with torch._dynamo.config.patch(repro_after="aot", repro_level=2):
            with patch(
                "torch.accelerator.synchronize", side_effect=lambda: called.append(1)
            ):
                compiled_fn = wrapped(gm, inputs)
                compiled_fn(inputs)

        self.assertGreater(len(called), 0)

    def test_inductor_fails_syncs_on_accelerator_tensor(self, device):
        # [FUNCTION DOC] Skips on CPU. With compile_fx_inner and accelerator synchronize
        # patched by fakes, runs inductor_fails on a traced tuple-output graph and asserts
        # synchronize was called.
        if torch.device(device).type == "cpu":
            self.skipTest("sync only triggered for non-CPU devices")

        from torch._dynamo.repro.after_aot import inductor_fails

        called = []

        def fake_compile_fx_inner(gm, args, **kwargs):
            # [FUNCTION DOC] HELPER: Fake compile_fx_inner returning a callable that runs
            # the graph module.
            def compiled(real_args):
                # [FUNCTION DOC] HELPER: Runs the graph module on the given real
                # arguments.
                return gm(*real_args)

            return compiled

        gm = torch.fx.symbolic_trace(TupleOut())
        args = [torch.randn(4, device=device)]

        with patch(
            "torch._inductor.compile_fx.compile_fx_inner", fake_compile_fx_inner
        ):
            with patch(
                "torch.accelerator.synchronize", side_effect=lambda: called.append(1)
            ):
                inductor_fails(gm, args)

        self.assertGreater(len(called), 0)

    def test_repro_run_syncs_on_accelerator_tensor(self, device):
        # [FUNCTION DOC] Skips on CPU. Patches repro_common to return a traced graph and
        # device args, compile_fx_inner and accelerator synchronize by fakes, runs repro_run
        # with a namespace with empty accuracy, and asserts synchronize was called.
        if torch.device(device).type == "cpu":
            self.skipTest("sync only triggered for non-CPU devices")

        import argparse

        from torch._dynamo.repro.after_aot import repro_run

        called = []
        gm = torch.fx.symbolic_trace(TupleOut())
        args = [torch.randn(4, device=device)]

        def fake_compile_fx_inner(gm, compile_args, **kwargs):
            # [FUNCTION DOC] HELPER: Fake compile_fx_inner returning a callable that runs
            # the graph module and wraps the result in a list.
            def compiled(real_args):
                # [FUNCTION DOC] HELPER: Runs the graph module on the given arguments,
                # returns the result in a list.
                return [gm(*real_args)]

            return compiled

        with patch(
            "torch._dynamo.repro.after_aot.repro_common", return_value=(gm, args)
        ):
            with patch(
                "torch._inductor.compile_fx.compile_fx_inner", fake_compile_fx_inner
            ):
                with patch(
                    "torch.accelerator.synchronize",
                    side_effect=lambda: called.append(1),
                ):
                    repro_run(argparse.Namespace(accuracy=""), None, None)

        self.assertGreater(len(called), 0)


instantiate_device_type_tests(TestWrapCompilerDebugSync, globals(), allow_xpu=True)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
