# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] Tests that scripted and lite-interpreter modules serialized with old operator
# versions (fixture files for versions two, seven and eight, under the test directory) behave like
# the current implementation: division (tensor, in-place, out, scalar, reciprocal, scalar-scalar),
# linspace and logspace, with and without output arguments. Uses hypothesis to generate inputs for
# the division tests; a missing fixture skips the test. Importing appends the test directory to
# sys.path; direct execution redirects to test/test_jit.py.

import io
import os
import sys
from itertools import product
from typing import Union

import hypothesis.strategies as st
from hypothesis import example, given, settings

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestSaveLoadForOpVersion(JitTestCase):
    # [CLASS DOC] Operator-version compatibility tests. Provides helpers to round-trip a module
    # through the standard format or the lite-interpreter format, to capture an exception as a
    # value, and to count graph nodes of a kind; each test compares a historic fixture module with
    # a Python reimplementation of the old behavior and a freshly saved current module.
    # Helper that returns the module after saving and loading
    def _save_load_module(self, m):
        # [FUNCTION DOC] HELPER: Instantiates the given module class, scripts it, saves it to an
        # in-memory buffer, rewinds the buffer, and returns the module loaded back with the
        # standard loader.
        scripted_module = torch.jit.script(m())
        buffer = io.BytesIO()
        torch.jit.save(scripted_module, buffer)
        buffer.seek(0)
        return torch.jit.load(buffer)

    def _save_load_mobile_module(self, m):
        # [FUNCTION DOC] HELPER: Instantiates the given module class, scripts it, serializes it
        # for the lite interpreter into an in-memory buffer, and returns the module loaded with
        # the lite-interpreter loader.
        scripted_module = torch.jit.script(m())
        buffer = io.BytesIO(scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        return _load_for_lite_interpreter(buffer)

    # Helper which returns the result of a function or the exception the
    #   function threw.
    def _try_fn(self, fn, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the function with the given arguments and returns its
        # result, or returns the raised exception object instead of propagating it.
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return e

    def _verify_no(self, kind, m):
        # [FUNCTION DOC] HELPER: Asserts that the module graph has no nodes of the given kind (a
        # count of zero).
        self._verify_count(kind, m, 0)

    def _verify_count(self, kind, m, count):
        # [FUNCTION DOC] HELPER: Counts how many times the kind text appears in the printed form
        # of all nodes of the module graph and asserts the total equals the expected count.
        node_count = sum(str(n).count(kind) for n in m.graph.nodes())
        self.assertEqual(node_count, count)

    """
    Tests that verify Torchscript remaps aten::div(_) from versions 0-3
    to call either aten::true_divide(_), if an input is a float type,
    or truncated aten::divide(_) otherwise.
    NOTE: currently compares against current div behavior, too, since
      div behavior has not yet been updated.
    """

    @settings(
        max_examples=10, deadline=200000
    )  # A total of 10 examples will be generated
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # Generate a pair (integer, float)
    @example((2, 3, 2.0, 3.0))  # Ensure this example will be covered
    def test_versioned_div_tensor(self, sample_input):
        # [FUNCTION DOC] Division of two tensors by operator, function and method forms. Compares
        # the version-two fixture against a reimplementation of historic division (true division
        # if either input is floating point, otherwise truncated division) and the current module
        # against the current division function. Hypothesis supplies pairs of an integer and a
        # float (ten examples) plus one explicit example; every pair element is paired with every
        # other. Skipped if the fixture cannot be loaded.
        def historic_div(self, other):
            # [FUNCTION DOC] HELPER: Historic division semantics for two tensors: true division if
            # either is floating point, otherwise truncating division.
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide(other)
            return self.divide(other, rounding_mode="trunc")

        # Tensor x Tensor
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing division three ways and returning all results.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Divides with the operator, the function and the method
                # form and returns the three results.
                result_0 = a / b
                result_1 = torch.div(a, b)
                result_2 = a.div(b)

                return result_0, result_1, result_2

        # Loads historic module
        try:
            v3_mobile_module = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_tensor_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module = self._save_load_mobile_module(MyModule)

        for val_a, val_b in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            def _helper(m, fn):
                # [FUNCTION DOC] HELPER: Runs the module and the reference function on the shared
                # inputs. If the module raised, asserts the reference also raised; otherwise
                # asserts every module output equals the reference result.
                m_results = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)

                if isinstance(m_results, Exception):
                    self.assertTrue(isinstance(fn_result, Exception))
                else:
                    for result in m_results:
                        self.assertEqual(result, fn_result)

            _helper(v3_mobile_module, historic_div)
            _helper(current_mobile_module, torch.div)

    @settings(
        max_examples=10, deadline=200000
    )  # A total of 10 examples will be generated
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # Generate a pair (integer, float)
    @example((2, 3, 2.0, 3.0))  # Ensure this example will be covered
    def test_versioned_div_tensor_inplace(self, sample_input):
        # [FUNCTION DOC] In-place division of tensors. Compares the version-two fixture with
        # historic in-place division and the current module with the current in-place method,
        # requiring equal results and that the module result equals the mutated input. Input a is
        # recreated between the two comparisons because it is modified. Hypothesis supplies pairs
        # of an integer and a float (ten examples) plus one explicit example; every pair element
        # is paired with every other. Skipped if the fixture cannot be loaded.
        # KNOWN ISSUE: the error branch uses assertTrue with the exception type as the message
        # argument, so it only checks the earlier result is truthy, not that it is an exception.
        def historic_div_(self, other):
            # [FUNCTION DOC] HELPER: Historic in-place division semantics: true division if either
            # input is floating point, otherwise truncating division, in place.
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide_(other)
            return self.divide_(other, rounding_mode="trunc")

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module dividing its first argument in place by its second and
            # returning it.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Applies in-place division and returns the first argument.
                a /= b
                return a

        try:
            v3_mobile_module = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_tensor_inplace_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module = self._save_load_mobile_module(MyModule)

        for val_a, val_b in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            def _helper(m, fn):
                # [FUNCTION DOC] HELPER: Runs the reference on a clone and the module on the
                # original input, then compares results and, on success, checks the module result
                # equals the mutated input.
                fn_result = self._try_fn(fn, a.clone(), b)
                m_result = self._try_fn(m, a, b)
                if isinstance(m_result, Exception):
                    self.assertTrue(fn_result, Exception)
                else:
                    self.assertEqual(m_result, fn_result)
                    self.assertEqual(m_result, a)

            _helper(v3_mobile_module, historic_div_)

            # Recreates a since it was modified in place
            a = torch.tensor((val_a,))
            _helper(current_mobile_module, torch.Tensor.div_)

    @settings(
        max_examples=10, deadline=200000
    )  # A total of 10 examples will be generated
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # Generate a pair (integer, float)
    @example((2, 3, 2.0, 3.0))  # Ensure this example will be covered
    def test_versioned_div_tensor_out(self, sample_input):
        # [FUNCTION DOC] Division with an output tensor, with a float and an integer output
        # buffer. Compares the version-two fixture against historic out-variant semantics and the
        # current module against torch.div, requiring equal results and that the result equals the
        # output buffer. Hypothesis supplies pairs of an integer and a float (ten examples) plus
        # one explicit example; every pair element is paired with every other. Skipped if the
        # fixture cannot be loaded.
        # KNOWN ISSUE: the error branch uses assertTrue with the exception type as the message
        # argument, so it only checks the earlier result is truthy, not that it is an exception.
        def historic_div_out(self, other, out):
            # [FUNCTION DOC] HELPER: Historic division semantics with an
            # output argument: true division if any of the three tensors is
            # floating point, otherwise truncating division.
            if (
                self.is_floating_point()
                or other.is_floating_point()
                or out.is_floating_point()
            ):
                return torch.true_divide(self, other, out=out)
            return torch.divide(self, other, out=out, rounding_mode="trunc")

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module dividing with an output argument.
            def forward(self, a, b, out):
                # [FUNCTION DOC] HELPER: Divides a by b writing into out and returns the result.
                return a.div(b, out=out)

        try:
            v3_mobile_module = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_tensor_out_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module = self._save_load_mobile_module(MyModule)

        for val_a, val_b in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            for out in (torch.empty((1,)), torch.empty((1,), dtype=torch.long)):

                def _helper(m, fn):
                    # [FUNCTION DOC] HELPER: Computes the reference result on a clone of the
                    # output buffer (keyword for torch.div, positional otherwise), runs the module
                    # on the real buffer, and compares results and the buffer.
                    fn_result = None
                    if fn is torch.div:
                        fn_result = self._try_fn(fn, a, b, out=out.clone())
                    else:
                        fn_result = self._try_fn(fn, a, b, out.clone())
                    m_result = self._try_fn(m, a, b, out)

                    if isinstance(m_result, Exception):
                        self.assertTrue(fn_result, Exception)
                    else:
                        self.assertEqual(m_result, fn_result)
                        self.assertEqual(m_result, out)

                _helper(v3_mobile_module, historic_div_out)
                _helper(current_mobile_module, torch.div)

    @settings(
        max_examples=10, deadline=200000
    )  # A total of 10 examples will be generated
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # Generate a pair (integer, float)
    @example((2, 3, 2.0, 3.0))  # Ensure this example will be covered
    def test_versioned_div_scalar(self, sample_input):
        # [FUNCTION DOC] Division of a tensor by a Python scalar, float and integer cases. The
        # float fixture module is compared with the current float module, the integer fixture with
        # historic integer semantics, and current modules with torch.div. Hypothesis supplies
        # pairs of an integer and a float (ten examples) plus one explicit example; every pair
        # element is paired with every other. Skipped if the fixture cannot be loaded.
        # KNOWN ISSUE: the error branch uses assertTrue with the exception type as the message
        # argument, so it only checks the earlier result is truthy, not that it is an exception.
        def historic_div_scalar_float(self, other: float):
            # [FUNCTION DOC] HELPER: Historic float scalar division: true division.
            return torch.true_divide(self, other)

        def historic_div_scalar_int(self, other: int):
            # [FUNCTION DOC] HELPER: Historic integer scalar division: true division for floating
            # tensors, otherwise truncating division.
            if self.is_floating_point():
                return torch.true_divide(self, other)
            return torch.divide(self, other, rounding_mode="trunc")

        class MyModuleFloat(torch.nn.Module):
            # [CLASS DOC] HELPER: Module dividing a tensor by a float.
            def forward(self, a, b: float):
                # [FUNCTION DOC] HELPER: Returns a divided by b.
                return a / b

        class MyModuleInt(torch.nn.Module):
            # [CLASS DOC] HELPER: Module dividing a tensor by an integer.
            def forward(self, a, b: int):
                # [FUNCTION DOC] HELPER: Returns a divided by b.
                return a / b

        try:
            v3_mobile_module_float = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/jit/fixtures/test_versioned_div_scalar_float_v2.ptl"
            )
            v3_mobile_module_int = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_int_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module_float = self._save_load_mobile_module(MyModuleFloat)
        current_mobile_module_int = self._save_load_mobile_module(MyModuleInt)

        for val_a, val_b in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = val_b

            def _helper(m, fn):
                # [FUNCTION DOC] HELPER: Runs the module and reference on the same tensor and
                # scalar; if the module raised, only checks the reference result is truthy,
                # otherwise asserts equal results.
                m_result = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)

                if isinstance(m_result, Exception):
                    self.assertTrue(fn_result, Exception)
                else:
                    self.assertEqual(m_result, fn_result)

            if isinstance(b, float):
                _helper(v3_mobile_module_float, current_mobile_module_float)
                _helper(current_mobile_module_float, torch.div)
            else:
                _helper(v3_mobile_module_int, historic_div_scalar_int)
                _helper(current_mobile_module_int, torch.div)

    @settings(
        max_examples=10, deadline=200000
    )  # A total of 10 examples will be generated
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # Generate a pair (integer, float)
    @example((2, 3, 2.0, 3.0))  # Ensure this example will be covered
    def test_versioned_div_scalar_reciprocal(self, sample_input):
        # [FUNCTION DOC] Division of a Python scalar by a tensor (reciprocal form). Fixture
        # modules for float and integer scalars are compared with current modules, and
        # current modules are compared with torch.div with swapped arguments. For integer
        # scalars against an integral tensor the comparison is skipped because the historic
        # behavior differed. Hypothesis supplies pairs of an integer and a float (ten
        # examples) plus one explicit example; every pair element is paired with every
        # other. Skipped if the fixture cannot be loaded.
        def historic_div_scalar_float_reciprocal(self, other: float):
            # [FUNCTION DOC] HELPER: Historic float reciprocal: the scalar divided by the tensor.
            return other / self

        def historic_div_scalar_int_reciprocal(self, other: int):
            # [FUNCTION DOC] HELPER: Historic integer reciprocal: true division for floating
            # tensors, otherwise truncating division of the scalar by the tensor.
            if self.is_floating_point():
                return other / self
            return torch.divide(other, self, rounding_mode="trunc")

        class MyModuleFloat(torch.nn.Module):
            # [CLASS DOC] HELPER: Module dividing a float by a tensor.
            def forward(self, a, b: float):
                # [FUNCTION DOC] HELPER: Returns b divided by a.
                return b / a

        class MyModuleInt(torch.nn.Module):
            # [CLASS DOC] HELPER: Module dividing an integer by a tensor.
            def forward(self, a, b: int):
                # [FUNCTION DOC] HELPER: Returns b divided by a.
                return b / a

        try:
            v3_mobile_module_float = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_reciprocal_float_v2.ptl"
            )
            v3_mobile_module_int = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_reciprocal_int_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module_float = self._save_load_mobile_module(MyModuleFloat)
        current_mobile_module_int = self._save_load_mobile_module(MyModuleInt)

        for val_a, val_b in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = val_b

            def _helper(m, fn):
                # [FUNCTION DOC] HELPER: Runs the module and computes the reference
                # (swapping argument order for torch.div); asserts equal results
                # unless the reference is a non-division function on an integral
                # tensor, in which case the comparison is skipped.
                m_result = self._try_fn(m, a, b)
                fn_result = None
                # Reverses argument order for torch.div
                if fn is torch.div:
                    fn_result = self._try_fn(torch.div, b, a)
                else:
                    fn_result = self._try_fn(fn, a, b)

                if isinstance(m_result, Exception):
                    self.assertTrue(isinstance(fn_result, Exception))
                elif fn is torch.div or a.is_floating_point():
                    self.assertEqual(m_result, fn_result)
                else:
                    # Skip when fn is not torch.div and a is integral because
                    # historic_div_scalar_int performs floored division
                    pass

            if isinstance(b, float):
                _helper(v3_mobile_module_float, current_mobile_module_float)
                _helper(current_mobile_module_float, torch.div)
            else:
                _helper(v3_mobile_module_int, current_mobile_module_int)
                _helper(current_mobile_module_int, torch.div)

    @settings(
        max_examples=10, deadline=200000
    )  # A total of 10 examples will be generated
    @given(
        sample_input=st.tuples(
            st.integers(min_value=5, max_value=199),
            st.floats(min_value=5.0, max_value=199.0),
        )
    )  # Generate a pair (integer, float)
    @example((2, 3, 2.0, 3.0))  # Ensure this example will be covered
    def test_versioned_div_scalar_inplace(self, sample_input):
        # [FUNCTION DOC] In-place division of a tensor by a scalar, using standard save/load
        # modules (not lite interpreter) and the in-place division method as reference. The
        # historic fixtures are loaded but only to skip the test when missing. Hypothesis supplies
        # pairs of an integer and a float (ten examples) plus one explicit example; every pair
        # element is paired with every other. Skipped if the fixture cannot be loaded.
        # KNOWN ISSUE: the error branch uses assertTrue with the exception type as the message
        # argument, so it only checks the earlier result is truthy, not that it is an exception.
        # KNOWN ISSUE: the module and reference both mutate and return the same tensor, so the
        # equality check is trivially true.
        def historic_div_scalar_float_inplace(self, other: float):
            # [FUNCTION DOC] HELPER: Historic in-place float
            # scalar division; unused by the checks.
            return self.true_divide_(other)

        def historic_div_scalar_int_inplace(self, other: int):
            # [FUNCTION DOC] HELPER: Historic in-place integer
            # scalar division; unused by the checks.
            if self.is_floating_point():
                return self.true_divide_(other)

            return self.divide_(other, rounding_mode="trunc")

        class MyModuleFloat(torch.nn.Module):
            # [CLASS DOC] HELPER: Module dividing a tensor in place by a float.
            def forward(self, a, b: float):
                # [FUNCTION DOC] HELPER: Applies in-place division and returns the tensor.
                a /= b
                return a

        class MyModuleInt(torch.nn.Module):
            # [CLASS DOC] HELPER: Module dividing a tensor in place by an integer.
            def forward(self, a, b: int):
                # [FUNCTION DOC] HELPER: Applies in-place division and returns the tensor.
                a /= b
                return a

        try:
            v3_mobile_module_float = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_inplace_float_v2.ptl"
            )
            v3_mobile_module_int = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_inplace_int_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module_float = self._save_load_module(MyModuleFloat)
        current_mobile_module_int = self._save_load_module(MyModuleInt)

        for val_a, val_b in product(sample_input, sample_input):
            a = torch.tensor((val_a,))
            b = val_b

            def _helper(m, fn):
                # [FUNCTION DOC] HELPER: Runs the module and the reference in-place method on the
                # same tensor and compares results.
                m_result = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)

                if isinstance(m_result, Exception):
                    self.assertTrue(fn_result, Exception)
                else:
                    self.assertEqual(m_result, fn_result)

            if isinstance(b, float):
                _helper(current_mobile_module_float, torch.Tensor.div_)
            else:
                _helper(current_mobile_module_int, torch.Tensor.div_)

    # NOTE: Scalar division was already true division in op version 3,
    #   so this test verifies the behavior is unchanged.
    def test_versioned_div_scalar_scalar(self):
        # [FUNCTION DOC] Scalar by scalar division with mixed int and float operands: the
        # version-two fixture and the current module must return the same four results. Scalar
        # division was already true division in op version three, so behavior is unchanged.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module dividing pairs of scalars and returning four quotients.
            def forward(self, a: float, b: int, c: float, d: int):
                # [FUNCTION DOC] HELPER: Computes four quotients of
                # the scalar inputs and returns them.
                result_0 = a / b
                result_1 = a / c
                result_2 = b / c
                result_3 = b / d
                return (result_0, result_1, result_2, result_3)

        try:
            v3_mobile_module = _load_for_lite_interpreter(
                pytorch_test_dir
                + "/cpp/jit/upgrader_models/test_versioned_div_scalar_scalar_v2.ptl"
            )
        except Exception as e:
            self.skipTest("Failed to load fixture!")

        current_mobile_module = self._save_load_mobile_module(MyModule)

        def _helper(m, fn):
            # [FUNCTION DOC] HELPER: Calls both modules with fixed scalar values and asserts every
            # output of the first equals the matching output of the second.
            vals = (5.0, 3, 2.0, 7)
            m_result = m(*vals)
            fn_result = fn(*vals)
            for mr, hr in zip(m_result, fn_result):
                self.assertEqual(mr, hr)

        _helper(v3_mobile_module, current_mobile_module)

    def test_versioned_linspace(self):
        # [FUNCTION DOC] Loads the version-seven linspace fixture, converts it to a
        # lite-interpreter module, and compares with a freshly saved current module on integer,
        # negative, float and complex inputs. The results with five steps and with one hundred
        # steps must have those lengths and equal the current results.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing linspace
            # results with five and one hundred steps.
            def forward(
                self, a: Union[int, float, complex], b: Union[int, float, complex]
            ):
                # [FUNCTION DOC] HELPER: Returns linspace between a and b with five steps and with
                # one hundred steps.
                c = torch.linspace(a, b, steps=5)
                d = torch.linspace(a, b, steps=100)
                return c, d

        scripted_module = torch.jit.load(
            pytorch_test_dir + "/jit/fixtures/test_versioned_linspace_v7.ptl"
        )

        buffer = io.BytesIO(scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v7_mobile_module = _load_for_lite_interpreter(buffer)

        current_mobile_module = self._save_load_mobile_module(Module)

        sample_inputs = ((3, 10), (-10, 10), (4.0, 6.0), (3 + 4j, 4 + 5j))
        for a, b in sample_inputs:
            (output_with_step, output_without_step) = v7_mobile_module(a, b)
            (current_with_step, current_without_step) = current_mobile_module(a, b)
            # when no step is given, should have used 100
            self.assertTrue(output_without_step.size(dim=0) == 100)
            self.assertTrue(output_with_step.size(dim=0) == 5)
            # outputs should be equal to the newest version
            self.assertEqual(output_with_step, current_with_step)
            self.assertEqual(output_without_step, current_without_step)

    def test_versioned_linspace_out(self):
        # [FUNCTION DOC] Like the linspace test but with an output tensor: loads the version-seven
        # fixture and compares output with the current module for int, float and complex inputs,
        # requiring length one hundred.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module writing linspace with one
            # hundred steps into an output tensor.
            def forward(
                self,
                a: Union[int, float, complex],
                b: Union[int, float, complex],
                out: torch.Tensor,
            ):
                # [FUNCTION DOC] HELPER: Returns linspace between a and b with one hundred steps
                # written into out.
                return torch.linspace(a, b, steps=100, out=out)

        model_path = (
            pytorch_test_dir + "/jit/fixtures/test_versioned_linspace_out_v7.ptl"
        )
        loaded_model = torch.jit.load(model_path)
        buffer = io.BytesIO(loaded_model._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v7_mobile_module = _load_for_lite_interpreter(buffer)
        current_mobile_module = self._save_load_mobile_module(Module)

        sample_inputs = (
            (
                3,
                10,
                torch.empty((100,), dtype=torch.int64),
                torch.empty((100,), dtype=torch.int64),
            ),
            (
                -10,
                10,
                torch.empty((100,), dtype=torch.int64),
                torch.empty((100,), dtype=torch.int64),
            ),
            (
                4.0,
                6.0,
                torch.empty((100,), dtype=torch.float64),
                torch.empty((100,), dtype=torch.float64),
            ),
            (
                3 + 4j,
                4 + 5j,
                torch.empty((100,), dtype=torch.complex64),
                torch.empty((100,), dtype=torch.complex64),
            ),
        )
        for start, end, out_for_old, out_for_new in sample_inputs:
            output = v7_mobile_module(start, end, out_for_old)
            output_current = current_mobile_module(start, end, out_for_new)
            # when no step is given, should have used 100
            self.assertTrue(output.size(dim=0) == 100)
            # "Upgraded" model should match the new version output
            self.assertEqual(output, output_current)

    def test_versioned_logspace(self):
        # [FUNCTION DOC] Same as the linspace test but for
        # logspace, using the version-eight fixture.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing logspace
            # results with five and one hundred steps.
            def forward(
                self, a: Union[int, float, complex], b: Union[int, float, complex]
            ):
                # [FUNCTION DOC] HELPER: Returns logspace between a and b with five steps and with
                # one hundred steps.
                c = torch.logspace(a, b, steps=5)
                d = torch.logspace(a, b, steps=100)
                return c, d

        scripted_module = torch.jit.load(
            pytorch_test_dir + "/jit/fixtures/test_versioned_logspace_v8.ptl"
        )

        buffer = io.BytesIO(scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v8_mobile_module = _load_for_lite_interpreter(buffer)

        current_mobile_module = self._save_load_mobile_module(Module)

        sample_inputs = ((3, 10), (-10, 10), (4.0, 6.0), (3 + 4j, 4 + 5j))
        for a, b in sample_inputs:
            (output_with_step, output_without_step) = v8_mobile_module(a, b)
            (current_with_step, current_without_step) = current_mobile_module(a, b)
            # when no step is given, should have used 100
            self.assertTrue(output_without_step.size(dim=0) == 100)
            self.assertTrue(output_with_step.size(dim=0) == 5)
            # outputs should be equal to the newest version
            self.assertEqual(output_with_step, current_with_step)
            self.assertEqual(output_without_step, current_without_step)

    def test_versioned_logspace_out(self):
        # [FUNCTION DOC] Same as the linspace output test but for logspace, using the
        # version-eight fixture.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module writing logspace with one
            # hundred steps into an output tensor.
            def forward(
                self,
                a: Union[int, float, complex],
                b: Union[int, float, complex],
                out: torch.Tensor,
            ):
                # [FUNCTION DOC] HELPER: Returns logspace between a and b with one hundred steps
                # written into out.
                return torch.logspace(a, b, steps=100, out=out)

        model_path = (
            pytorch_test_dir + "/jit/fixtures/test_versioned_logspace_out_v8.ptl"
        )
        loaded_model = torch.jit.load(model_path)
        buffer = io.BytesIO(loaded_model._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        v8_mobile_module = _load_for_lite_interpreter(buffer)
        current_mobile_module = self._save_load_mobile_module(Module)

        sample_inputs = (
            (
                3,
                10,
                torch.empty((100,), dtype=torch.int64),
                torch.empty((100,), dtype=torch.int64),
            ),
            (
                -10,
                10,
                torch.empty((100,), dtype=torch.int64),
                torch.empty((100,), dtype=torch.int64),
            ),
            (
                4.0,
                6.0,
                torch.empty((100,), dtype=torch.float64),
                torch.empty((100,), dtype=torch.float64),
            ),
            (
                3 + 4j,
                4 + 5j,
                torch.empty((100,), dtype=torch.complex64),
                torch.empty((100,), dtype=torch.complex64),
            ),
        )
        for start, end, out_for_old, out_for_new in sample_inputs:
            output = v8_mobile_module(start, end, out_for_old)
            output_current = current_mobile_module(start, end, out_for_new)
            # when no step is given, should have used 100
            self.assertTrue(output.size(dim=0) == 100)
            # "Upgraded" model should match the new version output
            self.assertEqual(output, output_current)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
