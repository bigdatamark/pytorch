# Owner(s): ["module: codegen"]
# [MODULE DOC] Tests for torchgen model parsing and ufunc codegen validation: error
# messages for invalid native_functions.yaml ufunc entries, custom class type parsing,
# alias Annotation parsing and BaseOperatorName parsing. No import-time side effects.

import textwrap
import unittest
from typing import cast

import expecttest
import yaml

import torchgen.dest as dest
import torchgen.gen as gen
from torchgen.gen import LineLoader, parse_native_yaml_struct
from torchgen.model import (
    Annotation,
    BaseOperatorName,
    CustomClassType,
    DispatchKey,
    NativeFunctionsGroup,
    Type,
)


class TestCodegenModel(expecttest.TestCase):
    # [CLASS DOC] Tests that malformed ufunc yaml definitions produce the expected parse or
    # codegen errors. Class attributes hold yaml snippet fragments for structured binary
    # and unary ops; indentation in them is hardcoded to two spaces.
    def assertParseErrorInline(self, yaml_str: str, expect: str) -> None:
        # [FUNCTION DOC] HELPER: Parses yaml text and expects the native function parser
        # to raise an assertion error; the message (context after the location marker
        # stripped, wrapped) is compared with the expected inline string. Fails the test
        # if nothing is raised.
        es = yaml.load(yaml_str, Loader=LineLoader)
        try:
            parse_native_yaml_struct(es, set())
        except AssertionError as e:
            # hack to strip out the context
            msg, _ = str(e).split("  in ", 2)
            self.assertExpectedInline("\n".join(textwrap.wrap(msg)), expect, skip=1)
            return
        self.fail(msg="Did not raise when expected to")

    def assertUfuncErrorInline(self, yaml_str: str, expect: str) -> None:
        # [FUNCTION DOC] HELPER: Parses yaml text that must contain exactly one structured
        # group with a ufunc inner loop (raises an assertion error otherwise), runs the
        # meta and native declaration generation as sanity checks, then expects the ufunc
        # cpu, cpu kernel and cuda generators to raise an assertion error whose stripped,
        # wrapped message equals the expected string. Fails the test if nothing is raised.
        # parse a single structured group out of the yaml to g
        es = yaml.load(yaml_str, Loader=LineLoader)
        parsed_yaml = parse_native_yaml_struct(es, set())
        native_functions, backend_indices = (
            parsed_yaml.native_functions,
            parsed_yaml.backend_indices,
        )
        grouped_native_functions = gen.get_grouped_native_functions(native_functions)
        if len(grouped_native_functions) != 1:
            raise AssertionError(
                f"Expected 1 grouped function, got {len(grouped_native_functions)}"
            )
        g = grouped_native_functions[0]
        if not isinstance(g, NativeFunctionsGroup):
            raise AssertionError(f"Expected NativeFunctionsGroup, got {type(g)}")
        if not g.out.ufunc_inner_loop:
            raise AssertionError("Expected g.out.ufunc_inner_loop to be truthy")
        # this is not ufunc codegen per se, but it does some basic sanity tests for
        # ufunc generation
        gen.compute_meta_function_declaration(g)
        dest.compute_native_function_declaration(g, backend_indices[DispatchKey.CPU])
        dest.compute_native_function_declaration(g, backend_indices[DispatchKey.CUDA])
        try:
            # the real kahuna
            dest.compute_ufunc_cpu(g)
            dest.compute_ufunc_cpu_kernel(g)
            dest.compute_ufunc_cuda(g)
        except AssertionError as e:
            # hack to strip out the context
            msg, _ = str(e).split("  in ", 2)
            self.assertExpectedInline("\n".join(textwrap.wrap(msg)), expect, skip=1)
            return
        self.fail(msg="Did not raise when expected to")

    # NB: indent is hardcoded to be two here, so format your yaml accordingly
    binop_out = (
        "func: binop.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"
    )
    ti_binop_out = f"""{binop_out}
  structured: True
  structured_inherits: TensorIteratorBase"""
    ti_binop = """func: binop(Tensor self, Tensor other) -> Tensor
  structured_delegate: binop.out
"""

    ti_unop_out = """func: unop.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase"""
    ti_unop = """func: unop(Tensor self) -> Tensor
  structured_delegate: unop.out
"""

    def test_nonstructured_ufunc(self) -> None:
        # [FUNCTION DOC] Asserts a ufunc inner loop on a non-structured function is
        # rejected with the message that ufunc must be structured.
        yaml_str = f"""\
- {self.binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
"""
        self.assertParseErrorInline(
            yaml_str,
            """\
ufunc must be structured""",
        )

    def test_overlapping_ufunc_and_dispatch(self) -> None:
        # [FUNCTION DOC] Asserts a ufunc function with an explicit CPU dispatch entry is
        # rejected with a message that ufunc should not have an explicit dispatch entry.
        yaml_str = f"""\
- {self.ti_binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
  dispatch:
    CPU: binop_cpu
"""
        self.assertParseErrorInline(
            yaml_str,
            """\
ufunc should not have explicit dispatch entry for CPU""",
        )

    # See https://github.com/pytorch/pytorch/pull/65851#discussion_r810238456
    @unittest.expectedFailure
    def test_scalaronly_shadowed(self) -> None:
        # [FUNCTION DOC] Expected-failure test: a ScalarOnly entry that duplicates the
        # Generic one should eventually raise a parse error, but currently no error (and
        # an empty expected message) is used.
        # KNOWN ISSUE: marked expectedFailure with an empty expected message; the
        # intended error is not implemented yet.
        yaml_str = f"""\
- {self.ti_binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
    ScalarOnly: binop (Bool)
"""
        self.assertParseErrorInline(
            yaml_str,
            """\
""",
        )

    def test_conflicting_ufunc(self) -> None:
        # [FUNCTION DOC] Asserts that ScalarOnly and Generic entries naming different ufunc
        # functions make ufunc codegen fail with a same-ufunc-name error.
        yaml_str = f"""\
- {self.ti_binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
    ScalarOnly: binop_scalar (Bool)
- {self.ti_binop}
"""
        self.assertUfuncErrorInline(
            yaml_str,
            """\
ScalarOnly and Generic must have same ufunc name""",
        )

    def test_invalid_cudafunctoronself_for_binary_op(self) -> None:
        # [FUNCTION DOC] Asserts that a CUDAFunctorOnSelf entry on a unary function makes
        # ufunc codegen fail with an error that it applies only to binary functions.
        yaml_str = f"""\
- {self.ti_unop_out}
  ufunc_inner_loop:
    Generic: unop (All)
    CUDAFunctorOnSelf: unop_self_cuda (All)
- {self.ti_unop}
"""
        self.assertUfuncErrorInline(
            yaml_str,
            """\
cannot use CUDAFunctorOnSelf on non-binary function""",
        )

    def test_parse_custom_class_type(self) -> None:
        # [FUNCTION DOC] Asserts that parsing a type string with the torch classes prefix
        # yields a custom class type whose class name excludes the prefix and whose string
        # form restores it.
        custom_class_name = "namespace_foo.class_bar"
        custom_class_name_with_prefix = f"__torch__.torch.classes.{custom_class_name}"
        custom_class_type = cast(
            CustomClassType, Type.parse(custom_class_name_with_prefix)
        )
        self.assertTrue(isinstance(custom_class_type, CustomClassType))
        self.assertEqual(custom_class_name, custom_class_type.class_name)
        self.assertEqual(custom_class_name_with_prefix, str(custom_class_type))


class TestAnnotation(expecttest.TestCase):
    # [CLASS DOC] Tests parsing of alias annotation strings into alias set, write flag and
    # post-write alias set. Stateless.
    def test_single_alias_no_write(self) -> None:
        # [FUNCTION DOC] Asserts a plain alias parses to one alias, not written, with an
        # empty after-set.
        a = Annotation.parse("a")
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertFalse(a.is_write)
        self.assertEqual(a.alias_set_after, ())

    def test_single_alias_is_write(self) -> None:
        # [FUNCTION DOC] Asserts an alias with the write marker parses as written with an
        # empty after-set.
        a = Annotation.parse("a!")
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, ())

    def test_single_alias_is_write_to_wildcard(self) -> None:
        # [FUNCTION DOC] Asserts a written alias mapped to the wildcard sets the after-set
        # to the wildcard.
        a = Annotation.parse("a! -> *")
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, tuple("*"))

    def test_alias_set(self) -> None:
        # [FUNCTION DOC] Asserts a pipe-separated alias list parses to a multi-element
        # alias set.
        a = Annotation.parse("a|b")
        self.assertEqual(a.alias_set, ("a", "b"))

    def test_alias_set_is_write_raises_exception(self) -> None:
        # [FUNCTION DOC] Asserts a multi-alias set with the write marker raises an
        # assertion error saying such a set is not mutable.
        with self.assertRaisesRegex(
            AssertionError, r"alias set larger than 1 is not mutable"
        ):
            Annotation.parse("a|b!")

    def test_single_alias_is_write_to_alias_set(self) -> None:
        # [FUNCTION DOC] Asserts a written alias mapped to a multi-alias set parses with
        # that set as the after-set.
        a = Annotation.parse("a! -> a|b")
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, ("a", "b"))

    def test_before_and_after_alias_set_larger_than_1_raises_exception(self) -> None:
        # [FUNCTION DOC] Asserts that multi-alias sets on both sides of the arrow raise an
        # assertion error about both sets being larger than one.
        with self.assertRaisesRegex(
            AssertionError,
            r"before alias set and after alias set cannot be larger than 1 at the same time",
        ):
            Annotation.parse("a|b -> c|d")


class TestBaseOperatorName(expecttest.TestCase):
    # [CLASS DOC] Tests parsing of base operator names. Stateless.
    def test_base_operator_name_with_ns_has_same_attributes_as_the_one_without_ns(
        self,
    ) -> None:
        # [FUNCTION DOC] Asserts parsing a namespaced dunder operator name gives the same
        # base, inplace and dunder method attributes as parsing it without the namespace.
        op = "aten::__lshift__"
        op_without_ns = "__lshift__"

        op_name = BaseOperatorName.parse(op)
        op_name_without_ns = BaseOperatorName.parse(op_without_ns)

        self.assertEqual(op_name.base, op_name_without_ns.base)
        self.assertEqual(op_name.inplace, op_name_without_ns.inplace)
        self.assertEqual(op_name.dunder_method, op_name_without_ns.dunder_method)


if __name__ == "__main__":
    unittest.main()
