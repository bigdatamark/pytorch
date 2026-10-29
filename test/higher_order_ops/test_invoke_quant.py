# Owner(s): ["module: higher order operators"]
# [MODULE DOC] Tests for the InvokeQuant higher-order op under eager, aot_eager and inductor
# backends: tracing, inline construction, multiple calls, graph pattern matching and prologue
# fusion. Creates a module-level InvokeQuant tracer instance at import. The base test class is
# deleted at the end so only the backend subclasses run.

import contextlib
import logging
import unittest

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from torch._higher_order_ops import InvokeQuant
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Ignored,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._inductor.utils import is_big_gpu, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    skipIfXpu,
    TestCase,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


invoke_quant_tracer = InvokeQuant()


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestInvokeQuant(TestCase):
    # [CLASS DOC] Base test class run once per backend; subclasses set the backend name.
    # Skipped under dynamo. Deleted from the module namespace after subclasses are defined.
    backend = ""

    def test_simple(self):
        # [FUNCTION DOC] Compiles a function calling the tracer with a quantization scheme and
        # options and checks the result equals the uncompiled reference.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns a one-element tuple with the product
            # of x and y plus y.
            return (torch.mul(x, y) + y,)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the tracer on gn with nf4 scheme and
            # returns the first output.
            return invoke_quant_tracer(
                gn, x, y, scheme="nf4", quant_options=invoke_quant_tracer
            )[0]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)
        ref = gn(x, y)[0]

        x_clone = x.clone().detach().requires_grad_(False)
        y_clone = y.clone().detach().requires_grad_(False)
        res = torch.compile(fn, backend=self.backend)(x_clone, y_clone)
        self.assertEqual(ref, res)

    def test_construct_inline(self):
        # [FUNCTION DOC] Same as the simple test but constructs InvokeQuant with low precision
        # codegen disabled; compiled result must match the reference.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns a one-element tuple with the product
            # of x and y plus y.
            return (torch.mul(x, y) + y,)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls a freshly constructed InvokeQuant on gn and returns
            # the first output.
            return InvokeQuant(codegen_low_precision=False)(gn, x, y, scheme="nf4")[0]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)
        ref = gn(x, y)[0]

        x_clone = x.clone().detach().requires_grad_(False)
        y_clone = y.clone().detach().requires_grad_(False)
        res = torch.compile(fn, backend=self.backend)(x_clone, y_clone)
        self.assertEqual(ref, res)

    def test_inline(self):
        # [FUNCTION DOC] Same as the simple test using a default InvokeQuant instance; compiled
        # result must match the reference.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns a one-element tuple with the product
            # of x and y plus y.
            return (torch.mul(x, y) + y,)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls a default InvokeQuant on gn and returns
            # the first output.
            return InvokeQuant()(gn, x, y, scheme="nf4")[0]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)
        ref = gn(x, y)[0]

        x_clone = x.clone().detach().requires_grad_(False)
        y_clone = y.clone().detach().requires_grad_(False)
        res = torch.compile(fn, backend=self.backend)(x_clone, y_clone)
        self.assertEqual(ref, res)

    def test_multiple(self):
        # [FUNCTION DOC] Enables post grad graph logging (global logging side
        # effect), compiles a function with two tracer calls and compares to
        # eager. For inductor, captures debug logs and checks the post grad graph
        # has two subgraphs each with an invoke_quant call using nf4.
        torch._logging.set_logs(post_grad_graphs=True)

        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns the product of x and y plus y.
            return torch.mul(x, y) + y

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Invokes the tracer twice on different inputs
            # and returns the sum.
            o1 = invoke_quant_tracer(gn, x, y, scheme="nf4")
            o2 = invoke_quant_tracer(gn, y, z, scheme="nf4")
            return o1 + o2

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)
        z = torch.randn(8, requires_grad=False)
        ref = fn(x, y, z)

        log_context = (
            contextlib.nullcontext()
            if self.backend != "inductor"
            else self.assertLogs(logger="torch._inductor", level=logging.DEBUG)
        )

        with log_context as log:
            res = torch.compile(fn, backend=self.backend)(x, y, z)

        self.assertEqual(ref, res)

        if self.backend == "inductor":
            logs = "\n".join(r.getMessage() for r in log.records)
            f = FileCheck()
            f.check("AFTER POST GRAD")
            f.check("subgraph0").check("subgraph1")
            for _ in range(2):
                f.check("torch.ops.higher_order.invoke_quant(").check_same("nf4")
            f.run(logs)


class TestInvokeQuantEager(TestInvokeQuant):
    # [CLASS DOC] Runs the base tests with the eager backend.
    backend = "eager"


class TestInvokeQuantAotEager(TestInvokeQuant):
    # [CLASS DOC] Runs the base tests with the aot_eager backend.
    backend = "aot_eager"


class TestInvokeQuantInductor(TestInvokeQuant):
    # [CLASS DOC] Runs the base tests with the inductor backend and adds pattern matching and
    # prologue fusion tests.
    backend = "inductor"

    def test_pattern_matching(self):
        # [FUNCTION DOC] Registers a graph pattern matching a matrix multiply of an
        # invoke_quant node with scheme nf4 in a custom post grad pre-pass (patched into
        # inductor config). Asserts the pattern handler fires once for the nf4 function and not
        # for one without a scheme.
        counter = 0

        test_pass = PatternMatcherPass()

        def my_pass(g):
            # [FUNCTION DOC] HELPER: Custom pass applying the test pattern pass to a graph.
            return test_pass.apply(g)

        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns the product of x and y plus y.
            return torch.mul(x, y) + y

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Matrix multiply of the nf4 tracer output with z.
            return invoke_quant_tracer(gn, x, y, scheme="nf4") @ z

        def fn_no_match(x, y, z):
            # [FUNCTION DOC] HELPER: Matrix multiply of the tracer output (no scheme) with z;
            # should not match.
            return invoke_quant_tracer(gn, x, y) @ z

        x = torch.randn(64, 64, requires_grad=False)
        y = torch.randn(64, 64, requires_grad=False)
        z = torch.randn(64, 64, requires_grad=False)

        @register_graph_pattern(
            CallFunction(
                torch.ops.aten.mm,
                CallFunction(
                    torch.ops.higher_order.invoke_quant,
                    Ignored(),
                    Ignored(),
                    Ignored(),
                    scheme="nf4",
                ),
                Arg(),
            ),
            pass_dict=test_pass,
        )
        def quant_matching(match: Match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Pattern handler that increments the enclosing
            # counter on each match.
            nonlocal counter
            counter += 1

        with unittest.mock.patch(
            "torch._inductor.config.post_grad_custom_pre_pass", my_pass
        ):
            torch.compile(fn)(x, y, z)
            self.assertTrue(counter == 1)

            torch.compile(fn_no_match)(x, y, z)
            self.assertTrue(counter == 1)

    @skipIfXpu(
        msg="MM Triton template fusion for XPU not work because the fusion"
        " can not speedup, unskip until #146568 fixed."
    )
    @requires_gpu()
    @config.patch(prologue_fusion=True)
    def test_prologue(self):
        # [FUNCTION DOC] Skipped without a large GPU. Compiles with max-autotune and prologue
        # fusion on float16 inputs (y filled with ones so numerics match), Triton GEMM backend
        # only. Checks the code has the k loop and a dot without float32 and that the result
        # equals the reference.
        if not is_big_gpu():
            raise unittest.SkipTest("requires large gpu to max-autotune")

        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns x times y plus y minus one.
            return torch.mul(x, y) + (y - 1)

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Matrix multiply of the tracer output on gn with z.
            return (
                invoke_quant_tracer(
                    gn, x, y, scheme="nf4", quant_options=invoke_quant_tracer
                )
                @ z
            )

        x = torch.randn(
            64, 64, requires_grad=False, device=GPU_TYPE, dtype=torch.float16
        )
        # make this a no-op to ensure equivalent numerics
        y = torch.randn(
            64, 64, requires_grad=False, device=GPU_TYPE, dtype=torch.float16
        ).fill_(1.0)
        z = torch.randn(
            64, 64, requires_grad=False, device=GPU_TYPE, dtype=torch.float16
        )
        ref = gn(x, y) @ z

        x_clone = x.clone().detach().requires_grad_(False)
        y_clone = y.clone().detach().requires_grad_(False)
        z_clone = z.clone().detach().requires_grad_(False)
        torch._dynamo.reset()
        with torch.no_grad(), config.patch(max_autotune_gemm_backends="TRITON"):
            fn_c = torch.compile(fn, mode="max-autotune-no-cudagraphs")
            res, code = run_and_get_code(fn_c, x_clone, y_clone, z_clone)

            FileCheck().check("k_idx in range").check_not("tl.float32").check(
                "tl.dot"
            ).run(code[0])

            self.assertEqual(ref, res)


del TestInvokeQuant

if __name__ == "__main__":
    run_tests()
