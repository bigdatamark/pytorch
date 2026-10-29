# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that inductor lowers functional custom ops that have a matching out-variant
# overload (tagged as out) into extern kernel calls to that out-variant, including multi-output
# ops, dynamic shapes, ops without an out-variant (which fall back to a regular fallback kernel),
# and the C++ wrapper runtime-dispatch fallback. Ops are registered in temporary fragment
# libraries named mylib. DEVICES lists cpu plus the GPU type when available.
"""
Tests for inductor lowering of functional custom ops to out-variant via ExternKernelOut.
"""

import torch
from torch._C import FileCheck
from torch._inductor import config
from torch._inductor.codegen import common
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


DEVICES = ("cpu", GPU_TYPE) if HAS_GPU else ("cpu",)


@instantiate_parametrized_tests
class TestCustomOpOutLowering(InductorTestCase):
    # [CLASS DOC] Parametrized test case (over device) for out-variant lowering of custom ops;
    # helper methods register ops into a scoped library passed by the test.
    """Tests for lowering functional custom ops to out-variant ExternKernelOut."""

    def _register_add_one_ops(self, lib):
        # [FUNCTION DOC] HELPER: Defines add_one (functional) and add_one.out (out-tagged)
        # schemas on the given library, registers composite implementations (add one, and copy
        # add one into the out tensor), plus a fake implementation. Returns the functional and
        # out overload objects.
        """Register a simple add_one op with functional + .out overloads."""
        lib.define("add_one(Tensor x) -> Tensor")
        lib.define(
            "add_one.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)",
            tags=(torch.Tag.out,),
        )

        def _add_one_impl(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Functional implementation; returns the input plus one.
            return x + 1

        def _add_one_out_impl(x: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Out implementation; writes the input plus one into the out
            # tensor and returns it.
            out.copy_(x + 1)
            return out

        lib.impl("add_one", _add_one_impl, "CompositeExplicitAutograd")
        lib.impl("add_one.out", _add_one_out_impl, "CompositeExplicitAutograd")

        @torch.library.register_fake("mylib::add_one", lib=lib)
        def _add_one_fake(x):
            # [FUNCTION DOC] HELPER: Fake implementation; returns an empty tensor of
            # the input's shape.
            return x.new_empty(x.shape)

        return torch.ops.mylib.add_one, torch.ops.mylib.add_one.out

    @parametrize("device", DEVICES)
    def test_add_one_lowered_to_out(self, device):
        # [FUNCTION DOC] Compiles a function calling add_one with fullgraph and checks
        # the result equals eager and the generated code calls the out overload and never
        # the default overload.
        """Test that a simple functional op gets lowered to its out-variant."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            self._register_add_one_ops(lib)

            def f(x):
                # [FUNCTION DOC] HELPER: Calls the add_one op on the input.
                return torch.ops.mylib.add_one(x)

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)

            FileCheck().check(".out(").check_not(".default(").run(code)

    @parametrize("device", DEVICES)
    def test_add_one_lowered_to_out_dynamic_shape(self, device):
        # [FUNCTION DOC] Same as the basic out-lowering test but marks the first dimension
        # dynamic and compiles with dynamic shapes, covering a symbolic output shape (issue
        # 185503). Checks equality with eager and out-variant call in the generated code.
        """Out-variant lowering with symbolic output shape (see #185503)."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            self._register_add_one_ops(lib)

            def f(x):
                # [FUNCTION DOC] HELPER: Calls the add_one op on the input.
                return torch.ops.mylib.add_one(x)

            x = torch.randn(4, 8, device=device)
            torch._dynamo.mark_dynamic(x, 0)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True, dynamic=True),
                x,
            )
            self.assertEqual(compiled_out, eager_out)
            FileCheck().check(".out(").check_not(".default(").run(code)

    def _register_split_add_ops(self, lib):
        # [FUNCTION DOC] HELPER: Defines split_add (functional, returns two tensors) and
        # split_add.out (two out tensors) on the library, with composite implementations
        # and a fake implementation returning two empty tensors. Returns the functional
        # and out overloads.
        """Register a split_add op returning two tensors with functional + .out overloads."""
        lib.define("split_add(Tensor x, float a, float b) -> (Tensor, Tensor)")
        lib.define(
            "split_add.out(Tensor x, float a, float b, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))",
            tags=(torch.Tag.out,),
        )

        def _split_add_impl(x, a, b):
            # [FUNCTION DOC] HELPER: Functional implementation; returns input plus a and
            # input plus b.
            return (x + a, x + b)

        def _split_add_out_impl(x, a, b, *, out0, out1):
            # [FUNCTION DOC] HELPER: Out implementation; writes input plus a and input plus b
            # into the two out tensors and returns them.
            out0.copy_(x + a)
            out1.copy_(x + b)
            return (out0, out1)

        lib.impl("split_add", _split_add_impl, "CompositeExplicitAutograd")
        lib.impl("split_add.out", _split_add_out_impl, "CompositeExplicitAutograd")

        @torch.library.register_fake("mylib::split_add", lib=lib)
        def _split_add_fake(x, a, b):
            # [FUNCTION DOC] HELPER: Fake implementation; returns two empty tensors of
            # the input's shape.
            return (x.new_empty(x.shape), x.new_empty(x.shape))

        return torch.ops.mylib.split_add, torch.ops.mylib.split_add.out

    @parametrize("device", DEVICES)
    def test_multi_output_lowered_to_out(self, device):
        # [FUNCTION DOC] Compiles a function using the two-output op, checks equality with eager,
        # and checks the generated code calls the out overload with both out arguments.
        """Test a two-output functional op gets lowered to its .out variant."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_split_add_ops(lib)

            def f(x):
                # [FUNCTION DOC] HELPER: Calls split_add and returns the sum of its two outputs.
                a, b = torch.ops.mylib.split_add(x, 1.0, 2.0)
                return a + b

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)
            FileCheck().check(".out(").check("out0=").check("out1=").run(code)

    @parametrize("device", DEVICES)
    def test_op_without_out_variant_falls_through(self, device):
        # [FUNCTION DOC] Registers an op with no out-variant and checks the compiled result
        # equals eager (the op falls through to the regular fallback kernel path).
        """Test that ops without an out-variant fall through to FallbackKernel."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("no_out_op(Tensor x) -> Tensor")

            def _impl(x):
                # [FUNCTION DOC] HELPER: Eager implementation; returns the input plus one.
                return x + 1

            lib.impl("no_out_op", _impl, "CompositeExplicitAutograd")

            @torch.library.register_fake("mylib::no_out_op", lib=lib)
            def _fake(x):
                # [FUNCTION DOC] HELPER: Fake implementation; returns an empty tensor of
                # the input's shape.
                return x.new_empty(x.shape)

            def f(x):
                # [FUNCTION DOC] HELPER: Calls the op without an out-variant.
                return torch.ops.mylib.no_out_op(x)

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)

    def test_cpp_wrapper_runtime_dispatch_single_output_fallback(self):
        # [FUNCTION DOC] CPU-only test of the C++ wrapper for an op lacking an out-variant.
        # Compiles with cpp wrapper, size asserts and caches disabled, checks equality with eager
        # and that the code uses the dispatcher call. Expects a size and stride assert for the
        # output except for the ArrayRef CPU wrapper (which must not emit it), and that no buffer
        # is assigned directly to another buffer.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("no_out_op(Tensor x) -> Tensor")

            def _impl(x):
                # [FUNCTION DOC] HELPER: Eager implementation; returns the input plus one.
                return x + 1

            lib.impl("no_out_op", _impl, "CompositeExplicitAutograd")

            @torch.library.register_fake("mylib::no_out_op", lib=lib)
            def _fake(x):
                # [FUNCTION DOC] HELPER: Fake implementation; returns an empty tensor of
                # the input's shape.
                return x.new_empty(x.shape)

            def f(x):
                # [FUNCTION DOC] HELPER: Calls the op without an out-variant.
                return torch.ops.mylib.no_out_op(x)

            x = torch.randn(4, 4)
            eager_out = f(x)

            with config.patch(
                cpp_wrapper=True, size_asserts=True, force_disable_caches=True
            ):
                compiled_out, code = run_and_get_code(
                    torch.compile(f, backend="inductor", fullgraph=True), x
                )
            self.assertEqual(compiled_out, eager_out)
            source_code = "\n".join(code)
            FileCheck().check("aoti_torch_call_dispatcher").run(source_code)
            output_assert = r'assert_size_stride\([^,]+,\s*\{4L?L?,\s*4L?L?\},\s*\{4L?L?,\s*1L?L?\},\s*"torch.ops.mylib.no_out_op.default"(, .*)?\)'
            wrapper_codegen = common.get_wrapper_codegen_for_device(
                "cpu", cpp_wrapper=True
            )
            if wrapper_codegen.__name__ == "CppWrapperCpuArrayRef":
                # ArrayRef wrapper tensors are not AtenTensorHandle, so that wrapper
                # path intentionally does not emit assert_size_stride.
                self.assertNotRegex(source_code, output_assert)
            else:
                FileCheck().check_regex(output_assert).run(source_code)
            self.assertNotRegex(source_code, r"\bbuf\d+\s*=\s*buf\d+\b")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
