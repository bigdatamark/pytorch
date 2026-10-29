# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that with the C++ wrapper enabled, calls to custom C++ ops registered through
# TORCH_LIBRARY are dispatched from generated C++ (schema lookup / boxed call or the AOTI
# dispatcher) instead of calling back into Python, and that Tensor-list returns work. Builds a C++
# extension inline at test time; no import-time side effects beyond imports.

import os
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE
from torch.testing._internal.inductor_utils import HAS_CPU


class TestCppWrapperCustomOps(InductorTestCase):
    # [CLASS DOC] Tests for custom-op code generation under the cpp_wrapper config.
    @staticmethod
    def _load_issue153478_extension():
        # [FUNCTION DOC] HELPER: compiles and loads, via load_inline, an extension (named with the
        # process id) that defines the issue153478_cpp_wrapper library with five CPU ops: a
        # tensor-list sum, a sum with a SymInt offset, an in-place scaled add returning nothing, a
        # mutate-and-return list op, and an op returning an optional tensor plus a tensor.
        # Compiled with -O0.
        from torch.utils.cpp_extension import load_inline

        cpp_src = r"""
        #include <ATen/ATen.h>
        #include <torch/library.h>

        at::Tensor sum_list(c10::List<at::Tensor> xs) {
            at::Tensor out = at::zeros_like(xs.get(0));
            for (const auto& x : xs) {
                out = out + x;
            }
            return out;
        }

        at::Tensor sum_list_with_n(c10::List<at::Tensor> xs, c10::SymInt n) {
            at::Tensor out = sum_list(xs);
            return out + n.expect_int();
        }

        void add_scaled_(at::Tensor& x, const at::Tensor& y, double scale) {
            x.add_(y, scale);
        }

        at::Tensor add_list_return(at::Tensor& x, c10::List<at::Tensor> xs) {
            for (const auto& y : xs) {
                x.add_(y);
            }
            return x + 1;
        }

        std::tuple<c10::optional<at::Tensor>, at::Tensor> optional_first(
            c10::List<at::Tensor> xs
        ) {
            return std::make_tuple(c10::nullopt, xs.get(0) + 1);
        }

        TORCH_LIBRARY(issue153478_cpp_wrapper, m) {
            m.def("sum_list(Tensor[] xs) -> Tensor");
            m.def("sum_list_with_n(Tensor[] xs, SymInt n) -> Tensor");
            m.def("add_scaled_(Tensor(a!) x, Tensor y, float scale) -> ()");
            m.def("add_list_return(Tensor(a!) x, Tensor[] xs) -> Tensor");
            m.def("optional_first(Tensor[] xs) -> (Tensor?, Tensor)");
        }

        TORCH_LIBRARY_IMPL(issue153478_cpp_wrapper, CPU, m) {
            m.impl("sum_list", sum_list);
            m.impl("sum_list_with_n", sum_list_with_n);
            m.impl("add_scaled_", add_scaled_);
            m.impl("add_list_return", add_list_return);
            m.impl("optional_first", optional_first);
        }
        """

        load_inline(
            name=f"issue153478_cpp_wrapper_ext_{os.getpid()}",
            cpp_sources=cpp_src,
            functions=[],
            extra_cflags=["-O0"],
            verbose=False,
        )

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "cpp_extension doesn't work here")
    @config.patch(cpp_wrapper=True)
    def test_cpp_custom_op_runtime_dispatch_stays_in_cpp(self):
        # [FUNCTION DOC] Registers fake implementations for the five extension ops, compiles five
        # small functions using them with fullgraph under cpp_wrapper and checks results against
        # eager math. For each, the generated code must not contain the Python callback markers
        # and must contain the dispatcher lookup (or AOTI dispatcher call for the op returning
        # nothing). Skipped without CPU, or under fbcode/sandcastle where cpp_extension does not
        # work.
        self._load_issue153478_extension()

        @torch.library.register_fake("issue153478_cpp_wrapper::sum_list")
        def _(xs):
            # [FUNCTION DOC] HELPER: fake kernel for sum_list; returns an empty tensor shaped like
            # the first input.
            return torch.empty_like(xs[0])

        @torch.library.register_fake("issue153478_cpp_wrapper::sum_list_with_n")
        def _(xs, n):
            # [FUNCTION DOC] HELPER: fake kernel for sum_list_with_n; returns an empty tensor like
            # the first input.
            return torch.empty_like(xs[0])

        @torch.library.register_fake("issue153478_cpp_wrapper::add_scaled_")
        def _(x, y, scale: float):
            # [FUNCTION DOC] HELPER: fake kernel for add_scaled_ (mutating, no result); returns
            # None.
            return None

        @torch.library.register_fake("issue153478_cpp_wrapper::add_list_return")
        def _(x, xs):
            # [FUNCTION DOC] HELPER: fake kernel for add_list_return; returns the mutated first
            # argument.
            return x

        @torch.library.register_fake("issue153478_cpp_wrapper::optional_first")
        def _(xs):
            # [FUNCTION DOC] HELPER: fake kernel for optional_first; returns None and an empty
            # tensor like the first list element.
            return None, torch.empty_like(xs[0])

        def tensor_list_fn(x, y):
            # [FUNCTION DOC] HELPER: calls the sum_list op on the two inputs packed into a list.
            return torch.ops.issue153478_cpp_wrapper.sum_list.default([x, y])

        def tensor_list_symint_fn(x, y):
            # [FUNCTION DOC] HELPER: calls sum_list_with_n on the two inputs with the first
            # dimension size as n.
            return torch.ops.issue153478_cpp_wrapper.sum_list_with_n.default(
                [x, y], x.shape[0]
            )

        def none_return_fn(x, y):
            # [FUNCTION DOC] HELPER: clones the first input, applies the in-place scaled add with
            # factor 2 using the second input, and returns the clone.
            z = x.clone()
            torch.ops.issue153478_cpp_wrapper.add_scaled_.default(z, y, 2.0)
            return z

        def mutation_and_return_fn(x, y):
            # [FUNCTION DOC] HELPER: clones the first input and returns the add_list_return op
            # applied to the clone and a list of the second input.
            z = x.clone()
            return torch.ops.issue153478_cpp_wrapper.add_list_return.default(z, [y])

        def optional_first_fn(x):
            # [FUNCTION DOC] HELPER: returns the second (tensor) output of optional_first for a
            # one-element list.
            return torch.ops.issue153478_cpp_wrapper.optional_first.default([x])[1]

        x = torch.randn(4)
        y = torch.randn(4)

        out, code = run_and_get_code(
            torch.compile(tensor_list_fn, fullgraph=True), x, y
        )
        self.assertEqual(out, x + y)
        code_str = "\n".join(code)
        self.assertNotIn("PyObject_CallObject", code_str)
        self.assertNotIn("custom_op_wrapper", code_str)
        self.assertIn(
            'findSchemaOrThrow("issue153478_cpp_wrapper::sum_list", "")',
            code_str,
        )
        self.assertIn("callBoxed", code_str)

        out, code = run_and_get_code(
            torch.compile(tensor_list_symint_fn, fullgraph=True), x, y
        )
        self.assertEqual(out, x + y + x.shape[0])
        code_str = "\n".join(code)
        self.assertNotIn("PyObject_CallObject", code_str)
        self.assertNotIn("custom_op_wrapper", code_str)
        self.assertIn(
            'findSchemaOrThrow("issue153478_cpp_wrapper::sum_list_with_n", "")',
            code_str,
        )
        self.assertIn("callBoxed", code_str)

        out, code = run_and_get_code(
            torch.compile(none_return_fn, fullgraph=True), x, y
        )
        self.assertEqual(out, x + 2 * y)
        code_str = "\n".join(code)
        self.assertNotIn("PyObject_CallObject", code_str)
        self.assertNotIn("custom_op_wrapper", code_str)
        self.assertIn(
            'aoti_torch_call_dispatcher("issue153478_cpp_wrapper::add_scaled_", ""',
            code_str,
        )

        out, code = run_and_get_code(
            torch.compile(mutation_and_return_fn, fullgraph=True), x, y
        )
        self.assertEqual(out, x + y + 1)
        code_str = "\n".join(code)
        self.assertNotIn("PyObject_CallObject", code_str)
        self.assertNotIn("custom_op_wrapper", code_str)
        self.assertIn(
            'findSchemaOrThrow("issue153478_cpp_wrapper::add_list_return", "")',
            code_str,
        )
        self.assertIn("callBoxed", code_str)

        out, code = run_and_get_code(
            torch.compile(optional_first_fn, fullgraph=True), x
        )
        self.assertEqual(out, x + 1)
        code_str = "\n".join(code)
        self.assertNotIn("PyObject_CallObject", code_str)
        self.assertNotIn("custom_op_wrapper", code_str)
        self.assertIn(
            'findSchemaOrThrow("issue153478_cpp_wrapper::optional_first", "")',
            code_str,
        )
        self.assertIn("callBoxed", code_str)

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @config.patch(cpp_wrapper=True, implicit_fallbacks=True)
    def test_cpp_wrapper_single_element_tensor_list_return(self):
        # [FUNCTION DOC] Defines in a scoped library an op returning a one-element tensor list
        # with CPU and Meta implementations, compiles a function that unpacks it, and checks the
        # numeric result and that the generated code uses the Python custom_op_wrapper fallback
        # and unwraps the list with PyList_GET_ITEM. Skipped without CPU.
        # A Tensor[]-returning op goes through the Python custom_op_wrapper
        # fallback, which always returns a Python list. A single-element result
        # must be unwrapped with PyList_GET_ITEM, not as a bare capsule (which
        # yields a null handle and segfaults downstream).
        with torch.library._scoped_library("mylib_single_list", "FRAGMENT") as m:
            m.define("single_list(Tensor x) -> Tensor[]")
            m.impl("single_list", lambda x: [x + 1], "CPU")
            m.impl("single_list", lambda x: [torch.empty_like(x)], "Meta")

            def fn(x):
                # [FUNCTION DOC] HELPER: unpacks the single tensor from the op result and
                # multiplies it by 2.
                (y,) = torch.ops.mylib_single_list.single_list.default(x)
                return y * 2

            x = torch.randn(4)
            out, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
            self.assertEqual(out, (x + 1) * 2)
            code_str = "\n".join(code)
            self.assertIn("custom_op_wrapper", code_str)
            self.assertIn("PyList_GET_ITEM", code_str)


if __name__ == "__main__":
    run_tests(needs="filelock")
