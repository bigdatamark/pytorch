# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the Inductor CuteDSL template backend (CuteDSLTemplate,
# CuteDSLTemplateKernel, CuteDSLOpOverrides, CuteDSLScheduling and the modification
# wrapper). Defines a CuteDSL elementwise-add template source string used by the end to
# end tests. The cutlass import is optional: a flag records availability, the codegen
# imports happen only when it is present, and the whole test class is skipped otherwise.
# Most tests run on mock graph handlers and inspect generated code text; two end to end
# tests need CUDA.
import unittest
from unittest.mock import MagicMock, patch

from expecttest import assert_expected_inline

import torch
from torch._inductor.test_case import TestCase
from torch._inductor.virtualized import V
from torch.testing._internal.inductor_utils import MockGraphHandler


try:
    import cutlass
    import cutlass.cute as cute  # noqa: F401

    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False

if HAS_CUTLASS:
    from torch._inductor.codegen.cutedsl.cutedsl_kernel import CuteDSLTemplateKernel
    from torch._inductor.codegen.cutedsl.cutedsl_scheduling import CuteDSLScheduling
    from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate
    from torch._inductor.select_algorithm import PartialRender


CUTEDSL_ADD_TEMPLATE = r"""
{{gen_defines()}}

@cute.kernel
def {{kernel_name}}_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx
    m, n = gA.shape

    if thread_idx < m * n:
        mi = thread_idx // n
        ni = thread_idx % n

        if mi < m and ni < n:
            gC[mi, ni] = gA[mi, ni] + gB[mi, ni]

@cute.jit
def {{kernel_name}}_jit(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream):
    {{gen_defines()}}
    m, n = mA.shape
    total_threads = m * n
    num_blocks = (total_threads + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    kernel = {{kernel_name}}_kernel(mA, mB, mC)
    kernel.launch(
        grid=[num_blocks, 1, 1],
        block=[THREADS_PER_BLOCK, 1, 1],
        stream=stream
    )

{{def_kernel("input_a", "input_b")}}
    cute_a = from_dlpack(input_a)
    cute_b = from_dlpack(input_b)
    cute_c = from_dlpack({{get_output()}})

    {{kernel_name}}_jit(cute_a, cute_b, cute_c, cuda.CUstream(stream))
    return {{get_output()}}
"""


@unittest.skipUnless(HAS_CUTLASS, "requires cutlass")
class TestCuteDSLTemplate(TestCase):
    # [CLASS DOC] Test case covering CuteDSL template kernels: import and define
    # generation, template registry uniqueness, def_kernel and get_output hooks, subgraph
    # lookup, index expression printing, scalar versus tensor SSA op lowering, load
    # fragment reuse, fast math flags, CSE and scheduler fusion. Skipped entirely when
    # cutlass cannot be imported. Holds no state.
    """Test cases for CuteDSL template functionality."""

    def test_vendored_dense_efc_kernel_configuration(self):
        # [FUNCTION DOC] Builds the vendored dense GEMM EFC kernel for a one-CTA and a
        # two-CTA configuration (tile M 128 with cluster M 1, and tile M 256 with cluster
        # M 2, tile N 64 in both) using float32 accumulation, bfloat16 output and the
        # identity epilogue converter. Asserts that arch is sm_100, the tiler and cluster
        # shapes and CTA group match the inputs, output dtype is bfloat16 and threads per
        # CTA is 224. Needs cutlass operators but no GPU.
        from cutlass.cute.nvgpu import tcgen05
        from cutlass.operators.providers.cutedsl.evt.converter import EFCConverter

        from torch._inductor.kernel.vendored_templates.cutedsl.dense_gemm_efc import (
            PersistentDenseGemmEFCKernel,
        )

        for use_2cta, tile_m, cluster_m, cta_group in (
            (False, 128, 1, tcgen05.CtaGroup.ONE),
            (True, 256, 2, tcgen05.CtaGroup.TWO),
        ):
            kernel = PersistentDenseGemmEFCKernel(
                cutlass.Float32,
                cutlass.BFloat16,
                use_2cta,
                (tile_m, 64),
                (cluster_m, 1),
                EFCConverter.identity_efc,
            )
            self.assertEqual(kernel.arch, "sm_100")
            self.assertEqual(kernel.mma_tiler_mn, (tile_m, 64))
            self.assertEqual(kernel.cluster_shape_mn, (cluster_m, 1))
            self.assertEqual(kernel.cta_group, cta_group)
            self.assertEqual(kernel.c_dtype, cutlass.BFloat16)
            self.assertEqual(kernel.threads_per_cta, 224)

    def test_gen_imports(self):
        # [FUNCTION DOC] Checks the import preamble produced by a CuteDSL kernel: it is a
        # string containing the torch, cutlass, cute and from_dlpack imports, has exactly
        # eight non-empty lines after stripping surrounding whitespace, and needs no graph
        # handler.
        kernel = CuteDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        imports = kernel.gen_imports()

        self.assertIn("import torch", imports)
        self.assertIn("import cutlass", imports)
        self.assertIn("import cutlass.cute as cute", imports)
        self.assertIn("from cutlass.cute.runtime import from_dlpack", imports)
        self.assertIsInstance(imports, str)

        lines = imports.strip().split("\n")
        self.assertEqual(len(lines), 8)

    def test_render_includes_imports(self):
        # [FUNCTION DOC] Renders a kernel using a mock template whose render returns a
        # fixed source with a cute kernel decorator. Asserts the result is a PartialRender
        # and that its code (ignoring leading whitespace) begins with the torch import,
        # includes the cutlass, cute and from_dlpack imports, and still contains the
        # original decorator text.
        template_source = """@cute.kernel
def {{kernel_name}}_kernel():
    pass

{{def_kernel("input", "output")}}
    return output"""

        mock_template = MagicMock()
        mock_template.render = MagicMock(return_value=template_source)

        kernel = CuteDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        result = kernel.render(mock_template)
        self.assertIsInstance(result, PartialRender)

        rendered_code = result._code

        # The imports might have leading whitespace, so strip it
        rendered_code_stripped = rendered_code.lstrip()

        self.assertTrue(
            rendered_code_stripped.startswith("import torch"),
            lambda msg: f"{msg}\nCode should start with 'import torch', got: {rendered_code_stripped[:50]}",
        )
        self.assertIn("import cutlass", rendered_code)
        self.assertIn("import cutlass.cute as cute", rendered_code)
        self.assertIn("from cutlass.cute.runtime import from_dlpack", rendered_code)
        self.assertIn("@cute.kernel", rendered_code)

    def test_template_env_contains_hooks(self):
        # [FUNCTION DOC] Verifies that the keyword arguments passed to the template render
        # call contain the def_kernel hook (callable) and the kernel_name entry. Uses a
        # local fake render that records its keyword arguments into a dict.
        kernel = CuteDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        captured_env = {}

        def mock_render(**kwargs):
            # [FUNCTION DOC] HELPER: Fake template render that copies all keyword
            # arguments into the enclosing captured_env mapping and returns the fixed
            # string "rendered".
            captured_env.update(kwargs)
            return "rendered"

        mock_template = MagicMock()
        mock_template.render = mock_render

        kernel.render(mock_template)

        self.assertIn("def_kernel", captured_env)
        self.assertIn("kernel_name", captured_env)
        self.assertTrue(callable(captured_env["def_kernel"]))

    def test_multiple_templates_unique_names(self):
        # [FUNCTION DOC] Registering two CuteDSLTemplate objects under the same name must
        # raise an AssertionError on the second. First removes any stale entry for the
        # per-instance test name from the class-level registry (global state shared across
        # tests).
        # Clean registry first
        test_name = f"unique_test_{id(self)}"
        if test_name in CuteDSLTemplate.all_templates:
            del CuteDSLTemplate.all_templates[test_name]

        _ = CuteDSLTemplate(
            name=test_name,
            source="template1",
        )

        with self.assertRaises(AssertionError):
            _ = CuteDSLTemplate(
                name=test_name,
                source="template2",
            )

    def test_indented_buffer_usage(self):
        # [FUNCTION DOC] Asserts that no non-empty line of the generated import preamble
        # starts with a space, i.e. the indented buffer used for imports emits them at
        # column zero.
        kernel = CuteDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        imports = kernel.gen_imports()

        lines = imports.strip().split("\n")
        for line in lines:
            if line:
                self.assertFalse(
                    line.startswith(" "),
                    lambda msg: f"{msg}\nLine should not be indented: '{line}'",
                )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cutedsl_add_e2e(self):
        # [FUNCTION DOC] End to end test, skipped without CUDA. Registers a template from
        # the add source and patches the lowering of the aten add op with a lowering that
        # builds one template choice (256 threads per block) and returns its output node,
        # falling back to the default lowering if building fails. Compiles an add function
        # with the inductor backend on 128 by 4 float32 CUDA tensors, checks the generated
        # code mentions cute and the THREADS_PER_BLOCK parameter, and checks the result
        # matches the eager sum within tolerance.
        """End-to-end test with CuteDSL template including code generation verification."""
        from torch._inductor.ir import TensorBox
        from torch._inductor.lowering import lowerings
        from torch._inductor.utils import run_and_get_code

        template = CuteDSLTemplate(
            name="test_add_e2e",
            source=CUTEDSL_ADD_TEMPLATE,
        )

        def cutedsl_add_lowering(a: TensorBox, b: TensorBox) -> TensorBox:
            # [FUNCTION DOC] HELPER: Replacement lowering for tensor add. Appends one
            # template choice with THREADS_PER_BLOCK 256; if appending reports an error or
            # yields no choice, delegates to the registered default add lowering, else
            # returns the choice's output node (no autotuning).
            choices = []
            error = template.maybe_append_choice(
                choices,
                input_nodes=[a, b],
                layout=a.get_layout(),
                THREADS_PER_BLOCK=256,
            )

            if error or not choices:
                default_lowering = lowerings[torch.ops.aten.add.Tensor]
                return default_lowering(a, b)

            # Use the single choice directly (no autotuning)
            return choices[0].output_node()

        with patch.dict(lowerings, {torch.ops.aten.add.Tensor: cutedsl_add_lowering}):
            # Test function
            def test_add(x, y):
                # [FUNCTION DOC] HELPER: Plain function adding its two tensor arguments;
                # the function that is compiled.
                return x + y

            device = "cuda"
            x = torch.randn(128, 4, device=device, dtype=torch.float32)
            y = torch.randn(128, 4, device=device, dtype=torch.float32)

            # Compile and get generated code
            compiled_fn = torch.compile(test_add, backend="inductor")
            result, (code,) = run_and_get_code(compiled_fn, x, y)

            # Verify CuteDSL code is present
            self.assertIn(
                "cute", code.lower(), "CuteDSL code should be in generated code"
            )
            # Verify parameter generation worked
            self.assertIn(
                "THREADS_PER_BLOCK", code, "Parameter should be in generated code"
            )

            # Verify correctness
            expected = x + y
            self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cutedsl_add_e2e_autotune(self):
        # [FUNCTION DOC] End to end autotune test, skipped without CUDA. Registers a
        # template and patches the add lowering to create one choice for each of 128, 256
        # and 512 threads per block (skipping variants that report an error), then selects
        # among them with autotune_select_algorithm, falling back to the default lowering
        # when there are no choices. Compiles and runs an add on 128 by 128 float32 CUDA
        # tensors and checks the result against the eager sum.
        """E2E test with multiple CuteDSL template variants for autotuning."""
        from torch._inductor.ir import TensorBox
        from torch._inductor.lowering import lowerings
        from torch._inductor.select_algorithm import autotune_select_algorithm

        template = CuteDSLTemplate(
            name="test_add_autotune",
            source=CUTEDSL_ADD_TEMPLATE,
        )

        def cutedsl_add_lowering(a: TensorBox, b: TensorBox) -> TensorBox:
            # [FUNCTION DOC] HELPER: Replacement add lowering that builds several thread
            # count variants of the template, autotunes among them and returns the
            # selected node; uses the default lowering if no variant could be built.
            choices = []

            # Add multiple variants with different thread counts for autotuning
            thread_variants = [128, 256, 512]
            for threads in thread_variants:
                error = template.maybe_append_choice(
                    choices,
                    input_nodes=[a, b],
                    layout=a.get_layout(),
                    THREADS_PER_BLOCK=threads,
                )
                if error:
                    # Skip this variant if it fails
                    continue

            if not choices:
                default_lowering = lowerings[torch.ops.aten.add.Tensor]
                return default_lowering(a, b)

            # Use autotuning to select the best variant
            node, _ = autotune_select_algorithm(
                "cutedsl_add_autotune",
                choices,
                [a, b],
                a.get_layout(),
            )
            return node

        with patch.dict(lowerings, {torch.ops.aten.add.Tensor: cutedsl_add_lowering}):
            # Test function
            def test_add(x, y):
                # [FUNCTION DOC] HELPER: Plain function adding its two tensor arguments;
                # the function that is compiled.
                return x + y

            device = "cuda"
            x = torch.randn(128, 128, device=device, dtype=torch.float32)
            y = torch.randn(128, 128, device=device, dtype=torch.float32)

            # Compile and run
            compiled_fn = torch.compile(test_add, backend="inductor")
            result = compiled_fn(x, y)

            # Verify correctness
            expected = x + y
            self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_gen_defines(self):
        # [FUNCTION DOC] Checks gen_defines output: keyword parameters become one
        # constexpr declaration line each, in argument order, for integers, booleans and
        # floats, compared with an inline expected string.
        """Test that gen_defines correctly generates CuteDSL parameter definitions."""
        kernel = CuteDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        # Test integer parameters
        params = kernel.gen_defines(
            THREADS_PER_BLOCK=256,
            BLOCK_SIZE=128,
            ENABLE_FEATURE=True,
        )

        assert_expected_inline(
            params,
            """\
THREADS_PER_BLOCK: cutlass.Constexpr = 256
BLOCK_SIZE: cutlass.Constexpr = 128
ENABLE_FEATURE: cutlass.Constexpr = True
""",
        )

        params_float = kernel.gen_defines(SCALE_FACTOR=1.5)
        assert_expected_inline(
            params_float,
            """\
SCALE_FACTOR: cutlass.Constexpr = 1.5
""",
        )

    def test_template_aliasing(self):
        # [FUNCTION DOC] Using mock buffers and a mock graph handler, creates a kernel
        # with two inputs and one output, calls def_kernel with custom names, and checks
        # the call returns the placeholder text and registers it in render_hooks. Invoking
        # the registered hook must yield code that aliases each custom name to the
        # corresponding argument name with an arg_ prefix.
        """Test that template variables are correctly aliased to function arguments."""
        from torch._inductor.ir import Buffer

        mock_input1 = MagicMock(spec=Buffer)
        mock_input1.get_name.return_value = "buf_input1"

        mock_input2 = MagicMock(spec=Buffer)
        mock_input2.get_name.return_value = "buf_input2"

        mock_output = MagicMock(spec=Buffer)
        mock_output.get_name.return_value = "buf_output"

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_aliasing",
                input_nodes=[mock_input1, mock_input2],
                output_node=mock_output,
            )

            def_kernel_hook = kernel.def_kernel("custom_a", "custom_b")
            self.assertEqual(def_kernel_hook, "<DEF_KERNEL>")

            self.assertIn("<DEF_KERNEL>", kernel.render_hooks)

            hook_fn = kernel.render_hooks["<DEF_KERNEL>"]
            generated_code = hook_fn()

            # Check that the generated code contains the expected aliasing statements
            self.assertIn("custom_a = arg_custom_a", generated_code)
            self.assertIn("custom_b = arg_custom_b", generated_code)

    def test_get_output_hook(self):
        # [FUNCTION DOC] Checks get_output: with a mock output buffer, it raises
        # ValueError while the kernel has no registered output buffer; after the buffer
        # name is mapped to an argument name in the kernel args, it returns that argument
        # name.
        """Test the get_output() template hook."""
        from torch._inductor.ir import Buffer

        mock_output = MagicMock(spec=Buffer)
        mock_output.get_name.return_value = "buf_test_output"

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_output",
                input_nodes=[],
                output_node=mock_output,
            )

            with self.assertRaises(ValueError):
                # error if no output buffer
                result = kernel.get_output()

            kernel.args.output_buffers["buf_test_output"] = "arg_buf_test_output"
            result = kernel.get_output()
            self.assertEqual(result, "arg_buf_test_output")

    def test_modification_subgraph(self):
        # [FUNCTION DOC] Checks subgraph lookup on a kernel built with two mock subgraphs:
        # index 0 and 1 return the matching subgraph and an out of range index raises
        # AssertionError.
        """Test the modification() method and subgraph processing."""

        from torch._inductor.ir import Buffer

        mock_subgraph1 = MagicMock(spec=Buffer)
        mock_subgraph2 = MagicMock(spec=Buffer)
        subgraphs = [mock_subgraph1, mock_subgraph2]

        mock_output = MagicMock(spec=Buffer)
        mock_output.get_name.return_value = "buf_output"

        kernel = CuteDSLTemplateKernel(
            kernel_name="test_modification",
            input_nodes=[],
            output_node=mock_output,
            subgraphs=subgraphs,
        )

        result = kernel._get_subgraph(0)
        self.assertEqual(result, mock_subgraph1)

        result = kernel._get_subgraph(1)
        self.assertEqual(result, mock_subgraph2)

        with self.assertRaises(AssertionError):
            kernel._get_subgraph(2)

    def test_kexpr_prints_modular_indexing(self):
        # [FUNCTION DOC] Verifies that kexpr prints a ModularIndexing sympy expression
        # with base 1 and modulus 2 as a parenthesized modulo of the symbol by 2.
        import sympy

        from torch.utils._sympy.functions import ModularIndexing

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_kexpr",
                input_nodes=[],
                output_node=None,
            )

            expr = ModularIndexing(sympy.Symbol("tmp1", integer=True), 1, 2)
            self.assertEqual(kernel.kexpr(expr), "(tmp1 % 2)")

    def test_index_expr_supports_modular_indexing(self):
        # [FUNCTION DOC] Verifies that the CuteDSL index_expr override accepts
        # ModularIndexing: it returns a CSEVariable with int64 dtype and the emitted
        # kernel body contains the modulo expression.
        import sympy

        from torch._inductor.codegen.common import CSEVariable
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )
        from torch.utils._sympy.functions import ModularIndexing

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_index_expr",
                input_nodes=[],
                output_node=None,
            )
            with V.set_kernel_handler(kernel):
                result = CuteDSLOpOverrides.index_expr(
                    ModularIndexing(sympy.Symbol("tmp1", integer=True), 1, 2),
                    torch.int64,
                )

            self.assertIsInstance(result, CSEVariable)
            self.assertEqual(result.dtype, torch.int64)
            self.assertIn("(tmp1 % 2)", kernel.body.getvalue())

    def test_index_expr_scalar_arithmetic_stays_scalar(self):
        # [FUNCTION DOC] Checks that index_expr results combined with the neg and add
        # overrides remain scalar: results are CSEVariable objects, the index_expr
        # attribute tracks the symbolic expression (including the added constant), the
        # body holds plain scalar assignments for the modulo, negation and addition, and
        # the body has no TensorSSA construction, full_like, shape or dtype usage.
        import sympy

        from torch._inductor.codegen.common import CSEVariable
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )
        from torch.utils._sympy.functions import ModularIndexing

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_index_expr_scalar_arithmetic",
                input_nodes=[],
                output_node=None,
            )
            with V.set_kernel_handler(kernel):
                idx = CuteDSLOpOverrides.index_expr(
                    ModularIndexing(sympy.Symbol("idx_in", integer=True), 1, 2),
                    torch.int64,
                )
                neg_result = CuteDSLOpOverrides.neg(idx)
                add_result = CuteDSLOpOverrides.add(idx, 1)

            self.assertIsInstance(idx, CSEVariable)
            self.assertIsInstance(neg_result, CSEVariable)
            self.assertIsInstance(add_result, CSEVariable)
            self.assertEqual(
                str(getattr(idx, "index_expr", None)), "ModularIndexing(idx_in, 1, 2)"
            )
            self.assertEqual(
                str(getattr(add_result, "index_expr", None)),
                "(ModularIndexing(idx_in, 1, 2)) + 1",
            )
            body = kernel.body.getvalue()
            self.assertIn("tmp0 = (idx_in % 2)", body)
            self.assertIn("tmp1 = (-tmp0)", body)
            self.assertIn("tmp2 = (tmp0 + 1)", body)
            self.assertNotIn("cute.TensorSSA(", body)
            self.assertNotIn("cute.full_like(", body)
            self.assertNotIn(".shape", body)
            self.assertNotIn(".dtype", body)

    def test_index_expr_abs_and_where_stay_scalar(self):
        # [FUNCTION DOC] Checks that abs, eq and where applied to a scalar index_expr
        # value stay scalar: all results are CSEVariable objects, the body uses the mlir
        # absi math call and a where call, and has no TensorSSA construction, full_like,
        # shape or dtype usage.
        import sympy

        from torch._inductor.codegen.common import CSEVariable
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )
        from torch.utils._sympy.functions import ModularIndexing

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_index_expr_abs_where",
                input_nodes=[],
                output_node=None,
            )
            with V.set_kernel_handler(kernel):
                idx = CuteDSLOpOverrides.index_expr(
                    ModularIndexing(sympy.Symbol("idx_in", integer=True), 1, 2),
                    torch.int64,
                )
                abs_result = CuteDSLOpOverrides.abs(idx)
                cond = CuteDSLOpOverrides.eq(idx, 0)
                where_result = CuteDSLOpOverrides.where(cond, idx, 1)

            self.assertIsInstance(idx, CSEVariable)
            self.assertIsInstance(abs_result, CSEVariable)
            self.assertIsInstance(cond, CSEVariable)
            self.assertIsInstance(where_result, CSEVariable)
            body = kernel.body.getvalue()
            self.assertIn("tmp0 = (idx_in % 2)", body)
            self.assertIn("mlir_math.absi(tmp0)", body)
            self.assertIn("cute.where(", body)
            self.assertNotIn("cute.TensorSSA(", body)
            self.assertNotIn("cute.full_like(", body)
            self.assertNotIn(".shape", body)
            self.assertNotIn(".dtype", body)

    def test_mod_does_not_attach_python_mod_index_expr(self):
        # [FUNCTION DOC] Checks that applying the mod override to an index_expr of a plain
        # symbol returns a value without an index_expr attribute (no symbolic expression
        # is attached).
        import sympy

        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_mod_index_expr",
                input_nodes=[],
                output_node=None,
            )
            with V.set_kernel_handler(kernel):
                idx = CuteDSLOpOverrides.index_expr(
                    sympy.Symbol("idx", integer=True), torch.int64
                )
                mod_result = CuteDSLOpOverrides.mod(idx, 4)

            self.assertIsNone(getattr(mod_result, "index_expr", None))

    def test_missing_vector_load_config_is_not_lane_uniform(self):
        # [FUNCTION DOC] Checks the modification wrapper's index fragment analysis. With
        # an empty vector load config it returns three values: not lane uniform, not
        # vector, no extra info. With a vec size of 1 config that names the index, it
        # returns lane uniform true, the other flags unchanged.
        import sympy

        from torch._inductor.codegen.cutedsl.cutedsl_kernel import (
            ModificationWrapperCuteDSL,
        )

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_missing_vector_load_config",
                input_nodes=[],
                output_node=None,
            )
            handler = ModificationWrapperCuteDSL(kernel, 0, {}, None)
            self.assertEqual(
                handler._analyze_index_fragment(sympy.Symbol("idx", integer=True)),
                (False, False, None),
            )

            scalar_handler = ModificationWrapperCuteDSL(
                kernel,
                0,
                {
                    "vector_load_vec_size": 1,
                    "vector_load_index": "idx",
                    "vector_load_dim": -1,
                },
                None,
            )
            self.assertEqual(
                scalar_handler._analyze_index_fragment(
                    sympy.Symbol("idx", integer=True)
                ),
                (True, False, None),
            )

    def test_neg_on_loaded_tensor_cse_uses_tensor_ssa(self):
        # [FUNCTION DOC] With a mock float32 buffer in the graph handler, loads from it
        # via the modification wrapper and applies the neg override. Asserts the loaded
        # value is a CSEVariable of shape (1,), the body allocates a one element register
        # tensor of float32, and the negation is emitted by building a TensorSSA from the
        # negated operand with its shape and dtype.
        import sympy

        from torch._inductor.codegen.common import CSEVariable
        from torch._inductor.codegen.cutedsl.cutedsl_kernel import (
            ModificationWrapperCuteDSL,
        )
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )

        mock_buffer = MagicMock()
        mock_buffer.dtype = torch.float32
        mock_graph = MockGraphHandler({"buf0": mock_buffer})

        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_neg_loaded_tensor",
                input_nodes=[],
                output_node=None,
            )
            handler = ModificationWrapperCuteDSL(kernel, 0, {}, None)
            with V.set_kernel_handler(kernel):
                loaded = handler.load("buf0", sympy.Integer(0))
                neg_result = CuteDSLOpOverrides.neg(loaded)

            self.assertIsInstance(loaded, CSEVariable)
            self.assertEqual(loaded.shape, (1,))
            self.assertIsInstance(neg_result, CSEVariable)
            body = kernel.body.getvalue()
            self.assertIn("tmp0 = cute.make_rmem_tensor(1, cutlass.Float32)", body)
            self.assertIn("tmp2 = cute.TensorSSA(-tmp1, tmp1.shape, tmp1.dtype)", body)

    def test_repeated_captured_load_reuses_generated_fragments(self):
        # [FUNCTION DOC] Loads from captured mock buffers with the modification wrapper.
        # Loading the same buffer and index twice returns the identical variable, while a
        # different index or different buffer returns a new one. The body must contain the
        # register tensor allocation exactly three times, showing the repeated load
        # generated no extra fragment.
        import sympy

        from torch._inductor.codegen.cutedsl.cutedsl_kernel import (
            ModificationWrapperCuteDSL,
        )

        buffers = {}
        for name in ("buf0", "buf1"):
            buffer = MagicMock()
            buffer.dtype = torch.float32
            buffer.get_size.return_value = (16,)
            buffers[name] = buffer
        mock_graph = MockGraphHandler(buffers)

        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_repeated_captured_load",
                input_nodes=[],
                output_node=None,
            )
            handler = ModificationWrapperCuteDSL(kernel, 0, {}, None)
            with V.set_kernel_handler(kernel):
                first = handler.load("buf0", sympy.Integer(0))
                repeated = handler.load("buf0", sympy.Integer(0))
                different_index = handler.load("buf0", sympy.Integer(1))
                different_buffer = handler.load("buf1", sympy.Integer(0))

        self.assertIs(first, repeated)
        self.assertIsNot(first, different_index)
        self.assertIsNot(first, different_buffer)
        self.assertEqual(
            kernel.body.getvalue().count("cute.make_rmem_tensor(1, cutlass.Float32)"),
            3,
        )

    def test_cutedsl_op_overrides(self):
        # [FUNCTION DOC] Exercises many CuteDSLOpOverrides methods (arithmetic, unary and
        # binary math, floor, logical xor) on two mock CSEVariable operands named tensor_a
        # and tensor_b, with shape (32, 64) and float32 dtype, inside a mock graph and
        # kernel handler. Each result must be a CSEVariable and the kernel body must
        # contain the expected operator or cute math call text. Also checks the scalar-to-
        # tensor helper wraps a scalar string in a full_like call and passes a tensor
        # through unchanged.
        """Test the new CuteDSLOpOverrides class."""
        import torch
        from torch._inductor.codegen.common import CSEVariable
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )
        from torch.utils._sympy.value_ranges import ValueRanges

        mock_cse_a = MagicMock(spec=CSEVariable)
        mock_cse_a.__str__.return_value = "tensor_a"
        mock_cse_a.dtype = torch.float32
        mock_cse_a.bounds = ValueRanges.unknown()
        mock_cse_a.shape = (32, 64)

        mock_cse_b = MagicMock(spec=CSEVariable)
        mock_cse_b.__str__.return_value = "tensor_b"
        mock_cse_b.dtype = torch.float32
        mock_cse_b.bounds = ValueRanges.unknown()
        mock_cse_b.shape = (32, 64)

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_ops",
                input_nodes=[],
                output_node=None,
            )
            with V.set_kernel_handler(kernel):
                result = CuteDSLOpOverrides.add(mock_cse_a, mock_cse_b)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.mul(mock_cse_a, mock_cse_b)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.truediv(mock_cse_a, mock_cse_b)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.exp(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.sqrt(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.maximum(mock_cse_a, mock_cse_b)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.minimum(mock_cse_a, mock_cse_b)
                self.assertIsInstance(result, CSEVariable)

                # Unary math ops
                result = CuteDSLOpOverrides.rsqrt(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.exp2(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.log2(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.log10(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.tan(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.acos(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.asin(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.atan(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.floor(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                # Binary math ops
                result = CuteDSLOpOverrides.atan2(mock_cse_a, mock_cse_b)
                self.assertIsInstance(result, CSEVariable)

                # Logical ops
                result = CuteDSLOpOverrides.logical_xor(mock_cse_a, mock_cse_b)
                self.assertIsInstance(result, CSEVariable)

            body = kernel.body.getvalue()

            # Arithmetic ops
            self.assertIn("(tensor_a + tensor_b)", body)
            self.assertIn("(tensor_a * tensor_b)", body)
            self.assertIn("(tensor_a / tensor_b)", body)

            # Unary math ops
            self.assertIn("cute.math.sqrt(tensor_a)", body)
            self.assertIn("cute.math.rsqrt(tensor_a)", body)
            self.assertIn("cute.math.exp2(tensor_a)", body)
            self.assertIn("cute.math.log2(tensor_a)", body)
            self.assertIn("cute.math.log10(tensor_a)", body)
            self.assertIn("cute.math.tan(tensor_a)", body)
            self.assertIn("cute.math.acos(tensor_a)", body)
            self.assertIn("cute.math.asin(tensor_a)", body)
            self.assertIn("cute.math.atan(tensor_a)", body)
            self.assertIn("cute.math.floor(tensor_a)", body)

            # Binary math ops
            self.assertIn("cute.math.atan2(tensor_a, tensor_b)", body)

            # Logical ops
            self.assertIn("(tensor_a ^ tensor_b)", body)

        scalar_result = CuteDSLOpOverrides._ensure_tensor_ssa(
            "5.0", mock_cse_a, is_tensor=False
        )
        self.assertEqual(scalar_result, "cute.full_like(tensor_a, 5.0)")

        tensor_result = CuteDSLOpOverrides._ensure_tensor_ssa(
            mock_cse_a, mock_cse_b, is_tensor=True
        )
        self.assertEqual(tensor_result, "tensor_a")

    def test_cutedsl_op_overrides_fast_math(self):
        # [FUNCTION DOC] Runs a set of unary ops plus atan2 on mock operands with the fast
        # math context manager enabled and disabled and compares generated bodies. With
        # fast math on, every line that calls cute math (and at least one must exist)
        # carries the fastmath flag; with it off the flag never appears.
        from torch._inductor.codegen.common import CSEVariable
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
            use_cutedsl_fast_math,
        )
        from torch.utils._sympy.value_ranges import ValueRanges

        mock_cse_a = MagicMock(spec=CSEVariable)
        mock_cse_a.__str__.return_value = "tensor_a"
        mock_cse_a.dtype = torch.float32
        mock_cse_a.bounds = ValueRanges.unknown()
        mock_cse_a.shape = (32, 64)

        mock_cse_b = MagicMock(spec=CSEVariable)
        mock_cse_b.__str__.return_value = "tensor_b"
        mock_cse_b.dtype = torch.float32
        mock_cse_b.bounds = ValueRanges.unknown()
        mock_cse_b.shape = (32, 64)

        def generate(fast_math):
            # [FUNCTION DOC] HELPER: Builds a fresh kernel, activates it as kernel handler
            # together with the fast math setting, applies each listed unary op name and
            # atan2 to the mock operands (checking each returns a CSEVariable), and
            # returns the accumulated kernel body text.
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_fast_math",
                input_nodes=[],
                output_node=None,
            )
            with (
                V.set_kernel_handler(kernel),
                use_cutedsl_fast_math(fast_math),
            ):
                for op_name in (
                    "exp",
                    "exp2",
                    "sqrt",
                    "rsqrt",
                    "log",
                    "log2",
                    "log10",
                    "cos",
                    "sin",
                    "tan",
                    "acos",
                    "asin",
                    "atan",
                    "erf",
                    "floor",
                    "tanh",
                    "sigmoid",
                ):
                    result = getattr(CuteDSLOpOverrides, op_name)(mock_cse_a)
                    self.assertIsInstance(result, CSEVariable)
                result = CuteDSLOpOverrides.atan2(mock_cse_a, mock_cse_b)
                self.assertIsInstance(result, CSEVariable)
            return kernel.body.getvalue()

        with V.set_graph_handler(MockGraphHandler()):
            fast_math_body = generate(True)
            precise_body = generate(False)

        math_lines = [
            line for line in fast_math_body.splitlines() if "cute.math." in line
        ]
        self.assertGreater(len(math_lines), 0)
        self.assertTrue(all("fastmath=True" in line for line in math_lines))
        self.assertNotIn("fastmath=True", precise_body)

    def test_cse_integration(self):
        # [FUNCTION DOC] Checks the kernel's CSE object: it is a CSE instance with name
        # prefix tmp, and generating an expression through it into the body yields a
        # variable whose name starts with tmp.
        """Test CSE (Common Subexpression Elimination) integration."""
        from torch._inductor.codegen.common import CSE

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_cse",
                input_nodes=[],
                output_node=None,
            )

            self.assertIsInstance(kernel.cse, CSE)
            self.assertEqual(kernel.cse.name_prefix, "tmp")

            with V.set_kernel_handler(kernel):
                test_expr = "x"
                var = kernel.cse.generate(kernel.body, test_expr, dtype=None)
                self.assertTrue(str(var).startswith("tmp"))

    def test_can_fuse_horizontal_returns_false(self):
        # [FUNCTION DOC] Checks that CuteDSLScheduling, built with no scheduler, answers
        # False to horizontal fusion for any pair of nodes.
        sched = CuteDSLScheduling(scheduler=None)
        self.assertIs(sched.can_fuse_horizontal(MagicMock(), MagicMock()), False)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
