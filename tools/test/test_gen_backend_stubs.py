# Owner(s): ["module: codegen"]
# [MODULE DOC] Tests for torchgen.gen_backend_stubs.run, the entry point used by external
# backends: valid yaml inputs succeed and malformed ones (missing or invalid backend, missing
# namespace, bad supported list, supported/autograd conflicts, unknown keys, non-boolean flags,
# wrong kernel names) raise assertion errors with specific messages. Yaml is written to
# temporary files.

from __future__ import annotations

import tempfile
import unittest

import expecttest

from torchgen.gen import _GLOBAL_PARSE_NATIVE_YAML_CACHE
from torchgen.gen_backend_stubs import run


# gen_backend_stubs.py is an integration point that is called directly by external backends.
# The tests here are to confirm that badly formed inputs result in reasonable error messages.
class TestGenBackendStubs(expecttest.TestCase):
    # [CLASS DOC] Test case for the backend stub generator input validation. Stateless apart
    # from clearing the global parsed native yaml cache before each test.
    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Clears the module-level native yaml parse cache so tests do
        # not see each other's parsed data (mutates global state). The global declaration is
        # not needed since the cache is only mutated.
        global _GLOBAL_PARSE_NATIVE_YAML_CACHE
        _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()

    def assert_success_from_gen_backend_stubs(self, yaml_str: str) -> None:
        # [FUNCTION DOC] HELPER: Writes the yaml text to a temporary file and runs the stub
        # generator on it in dry-run mode with an empty output directory; any error
        # propagates and fails the test.
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(yaml_str)
            fp.flush()
            run(fp.name, "", True)

    def get_errors_from_gen_backend_stubs(
        self, yaml_str: str, *, kernels_str: str | None = None
    ) -> str:
        # [FUNCTION DOC] HELPER: Writes the yaml text (and optionally kernel source text, used
        # as the implementation path) to temporary files and runs the stub generator in
        # dry-run mode, expecting an assertion error. Returns the error text with the
        # temporary yaml file name removed; fails the test if no error is raised.
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(yaml_str)
            fp.flush()
            try:
                if kernels_str is None:
                    run(fp.name, "", True)
                else:
                    with tempfile.NamedTemporaryFile(mode="w") as kernel_file:
                        kernel_file.write(kernels_str)
                        kernel_file.flush()
                        run(fp.name, "", True, impl_path=kernel_file.name)
            except AssertionError as e:
                # Scrub out the temp file name from any error messages to simplify assertions.
                return str(e).replace(fp.name, "")
            self.fail(
                "Expected gen_backend_stubs to raise an AssertionError, but it did not."
            )

    def test_valid_single_op(self) -> None:
        # [FUNCTION DOC] Asserts a minimal yaml with one supported operator is accepted.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_multiple_ops(self) -> None:
        # [FUNCTION DOC] Asserts a yaml with several supported operators, one with an
        # overload name, is accepted.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- add.Tensor
- abs"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_zero_ops(self) -> None:
        # [FUNCTION DOC] Asserts a yaml whose supported list is empty is accepted.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:"""
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_zero_ops_doesnt_require_backend_dispatch_key(self) -> None:
        # [FUNCTION DOC] Asserts that with no supported operators even an unknown backend
        # name is accepted, since the backend is not parsed.
        yaml_str = """\
backend: BAD_XLA
cpp_namespace: torch_xla
supported:"""
        # External codegen on a yaml file with no operators is effectively a no-op,
        # so there's no reason to parse the backend
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_with_autograd_ops(self) -> None:
        # [FUNCTION DOC] Asserts a yaml with both supported and autograd operators in
        # separate operator groups is accepted.
        # KNOWN ISSUE: the in-body comment says the yaml has no operators, but it lists
        # operators under both supported and autograd; the comment was copied from another test.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs
autograd:
- add.Tensor"""
        # External codegen on a yaml file with no operators is effectively a no-op,
        # so there's no reason to parse the backend
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_missing_backend(self) -> None:
        # [FUNCTION DOC] Asserts omitting the backend key yields the error that a backend
        # value must be provided.
        yaml_str = """\
cpp_namespace: torch_xla
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error, '''You must provide a value for "backend"'''
        )

    def test_empty_backend(self) -> None:
        # [FUNCTION DOC] Asserts an empty backend value yields the same must-provide-backend
        # error as a missing one.
        yaml_str = """\
backend:
cpp_namespace: torch_xla
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error, '''You must provide a value for "backend"'''
        )

    def test_backend_invalid_dispatch_key(self) -> None:
        # [FUNCTION DOC] Asserts a backend that is not a valid dispatch key yields an error
        # naming the unknown key and saying it must be a valid dispatch key.
        yaml_str = """\
backend: NOT_XLA
cpp_namespace: torch_xla
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """\
unknown dispatch key NOT_XLA
  The provided value for "backend" must be a valid DispatchKey, but got NOT_XLA.""",
        )

    def test_missing_cpp_namespace(self) -> None:
        # [FUNCTION DOC] Asserts omitting the C++ namespace yields the must-provide error.
        yaml_str = """\
backend: XLA
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error, '''You must provide a value for "cpp_namespace"'''
        )

    def test_whitespace_cpp_namespace(self) -> None:
        # [FUNCTION DOC] Asserts a C++ namespace containing only whitespace yields the same
        # must-provide error.
        yaml_str = """\
backend: XLA
cpp_namespace:\t
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error, '''You must provide a value for "cpp_namespace"'''
        )

    # supported is a single item (it should be a list)
    def test_nonlist_supported(self) -> None:
        # [FUNCTION DOC] Asserts a supported value that is a plain string instead of a list
        # yields an error stating the expected list and the actual value and type.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported: abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """expected "supported" to be a list, but got: abs (of type <class 'str'>)""",
        )

    # supported contains an op that isn't in native_functions.yaml
    def test_supported_invalid_op(self) -> None:
        # [FUNCTION DOC] Asserts an operator name absent from the native functions yields an
        # invalid operator name error.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs_BAD"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error, """Found an invalid operator name: abs_BAD"""
        )

    # The backend is valid, but doesn't have a valid autograd key. They can't override autograd kernels in that case.
    # Only using Vulkan here because it has a valid backend key but not an autograd key- if this changes we can update the test.
    def test_backend_has_no_autograd_key_but_provides_entries(self) -> None:
        # [FUNCTION DOC] Asserts that for a backend lacking an autograd key (Vulkan) the
        # supported operator is reported as an invalid operator name.
        yaml_str = """\
backend: Vulkan
cpp_namespace: torch_vulkan
supported:
- add
autograd:
- sub"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error, """Found an invalid operator name: add"""
        )

    # in an operator group, currently all operators must either be registered to the backend or autograd kernel.
    # Here, functional and out mismatch
    def test_backend_autograd_kernel_mismatch_out_functional(self) -> None:
        # [FUNCTION DOC] Asserts splitting a functional and an out variant of one op between
        # the supported and autograd lists yields the cannot-mix-and-match error naming both.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- add.Tensor
autograd:
- add.out"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add_out is listed under "autograd".""",
        )

    # in an operator group, currently all operators must either be registered to the backend or autograd kernel.
    # Here, functional and inplace mismatch
    def test_backend_autograd_kernel_mismatch_functional_inplace(self) -> None:
        # [FUNCTION DOC] Asserts splitting a functional and an inplace variant between the
        # supported and autograd lists yields the same cannot-mix-and-match error.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- add.Tensor
autograd:
- add_.Tensor"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add_ is listed under "autograd".""",
        )

    # Currently, the same operator can't be listed under both 'supported' and 'autograd', which would
    # involve registering the same kernel to both the XLA and AutogradXLA keys.
    # If we need that functionality in the future, we'll need to augment the codegen.
    def test_op_appears_in_supported_and_autograd_lists(self) -> None:
        # [FUNCTION DOC] Asserts listing the same operator under both supported and autograd
        # yields the cannot-mix-and-match error.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- add.Tensor
autograd:
- add.Tensor"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add is listed under "autograd".""",
        )

    # unrecognized extra yaml key
    def test_unrecognized_key(self) -> None:
        # [FUNCTION DOC] Asserts an unknown top-level yaml key yields an error listing the
        # unexpected key and the full set of supported keys.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs
invalid_key: invalid_val"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """ contains unexpected keys: invalid_key. Only the following keys are supported: backend, class_name, cpp_namespace, extra_headers, supported, autograd, full_codegen, non_native, ir_gen, symint""",
        )

    # if use_out_as_primary is provided, it must be a bool
    def test_use_out_as_primary_non_bool(self) -> None:
        # [FUNCTION DOC] Asserts a non-boolean use_out_as_primary value yields an error
        # quoting the provided value.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
use_out_as_primary: frue
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """You must provide either True or False for use_out_as_primary. Provided: frue""",
        )

    # if device_guard is provided, it must be a bool
    def test_device_guard_non_bool(self) -> None:
        # [FUNCTION DOC] Asserts a non-boolean device_guard value yields an error quoting
        # the provided value.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
device_guard: frue
supported:
- abs"""
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(
            output_error,
            """You must provide either True or False for device_guard. Provided: frue""",
        )

    def test_incorrect_kernel_name(self) -> None:
        # [FUNCTION DOC] Supplies kernel source in which one expected kernel name is wrong
        # and asserts the error reports the missing kernel, the zero found versus one
        # expected count, and the expected function schema.
        yaml_str = """\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs
autograd:
- add.Tensor"""
        # Codegen will expect two kernel names (and try to parse them with regex):
        # XLANativeFunctions::abs(...)
        # XLANativeFunctions::add(...)
        kernels_str = """\
at::Tensor& XLANativeFunctions::absWRONG(at::Tensor& self) {}
at::Tensor& XLANativeFunctions::add(at::Tensor& self) {}"""
        output_error = self.get_errors_from_gen_backend_stubs(
            yaml_str, kernels_str=kernels_str
        )
        self.assertExpectedInline(
            output_error,
            """\

XLANativeFunctions is missing a kernel definition for abs. We found 0 kernel(s) with that name,
but expected 1 kernel(s). The expected function schemas for the missing operator are:
at::Tensor abs(const at::Tensor & self)

""",
        )


if __name__ == "__main__":
    unittest.main()
