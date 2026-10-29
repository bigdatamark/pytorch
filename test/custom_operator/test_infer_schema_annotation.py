# Owner(s): ["module: pt2-dispatcher"]
# [MODULE DOC] Tests for torch.library.infer_schema when the operator's parameters and return are
# plain Python type annotations. Each test builds small functions, infers the schema string and
# compares it with the expected schema text. Uses postponed annotations; Sequence is imported only
# under TYPE_CHECKING. Defines TestInferSchemaWithAnnotation.
from __future__ import annotations

import typing
import unittest

import torch
from torch import Tensor, types
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


if typing.TYPE_CHECKING:
    from collections.abc import Sequence


mutates_args = {}


@skipIfTorchDynamo("custom operator tests not applicable to dynamo")
class TestInferSchemaWithAnnotation(TestCase):
    # [CLASS DOC] Test cases for schema inference from annotations; skipped under dynamo. All tests
    # use the module-level empty mutates_args mapping.
    def test_tensor(self):
        # [FUNCTION DOC] Tensor parameters and returns map to Tensor in the inferred schema, for
        # one and for two arguments.
        def foo_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: returns a clone of x.
            return x.clone()

        result = torch.library.infer_schema(foo_op, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        def foo_op_2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: returns the clone of x plus y.
            return x.clone() + y

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x, Tensor y) -> Tensor")

    def test_native_types(self):
        # [FUNCTION DOC] int becomes SymInt, bool stays bool, str stays str and float stays float.
        def foo_op(x: int) -> int:
            # [FUNCTION DOC] HELPER: returns x unchanged.
            return x

        result = torch.library.infer_schema(foo_op, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt x) -> SymInt")

        def foo_op_2(x: bool) -> bool:
            # [FUNCTION DOC] HELPER: returns x unchanged.
            return x

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(bool x) -> bool")

        def foo_op_3(x: str) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_3, mutates_args=mutates_args)
        self.assertEqual(result, "(str x) -> SymInt")

        def foo_op_4(x: float) -> float:
            # [FUNCTION DOC] HELPER: returns x unchanged.
            return x

        result = torch.library.infer_schema(foo_op_4, mutates_args=mutates_args)
        self.assertEqual(result, "(float x) -> float")

    def test_torch_types(self):
        # [FUNCTION DOC] Number maps to Scalar, torch.dtype to ScalarType and torch.device to
        # Device in the inferred schema.
        def foo_op_1(x: torch.types.Number) -> torch.types.Number:
            # [FUNCTION DOC] HELPER: returns x unchanged.
            return x

        result = torch.library.infer_schema(foo_op_1, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

        def foo_op_2(x: torch.dtype) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(ScalarType x) -> SymInt")

        def foo_op_3(x: torch.device) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_3, mutates_args=mutates_args)
        self.assertEqual(result, "(Device x) -> SymInt")

    def test_type_variants(self):
        # [FUNCTION DOC] Optional ints, sequences and lists of ints (with and without optional) and
        # unions of int, float and bool in either order map to SymInt?, SymInt[], SymInt[]? and
        # Scalar.
        def foo_op_1(x: int | None) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_1, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt? x) -> SymInt")

        def foo_op_2(x: typing.Sequence[int]) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        def foo_op_3(x: list[int]) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_3, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        def foo_op_4(x: typing.Sequence[int] | None) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_4, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        def foo_op_5(x: list[int] | None) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_5, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        def foo_op_6(x: int | float | bool) -> types.Number:
            # [FUNCTION DOC] HELPER: returns x unchanged.
            return x

        result = torch.library.infer_schema(foo_op_6, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

        def foo_op_7(x: int | bool | float) -> types.Number:
            # [FUNCTION DOC] HELPER: returns x unchanged.
            return x

        result = torch.library.infer_schema(foo_op_7, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

    @unittest.expectedFailure
    def test_no_library_prefix(self):
        # [FUNCTION DOC] Repeats the schema checks using unqualified names (Tensor, Sequence, list,
        # union types). Marked as an expected failure; the Sequence name is imported only under
        # the type-checking guard, so it is unresolved at runtime.
        def foo_op(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: returns a clone of x.
            return x.clone()

        result = torch.library.infer_schema(foo_op, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        def foo_op_2(x: Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: returns a clone of x.
            return x.clone()

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        def foo_op_3(x: torch.Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: returns a clone of x.
            return x.clone()

        result = torch.library.infer_schema(foo_op_3, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        def foo_op_4(x: list[int]) -> types.Number:
            # [FUNCTION DOC] HELPER: returns the first element of x.
            return x[0]

        result = torch.library.infer_schema(foo_op_4, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> Scalar")

        def foo_op_5(x: int | None) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_5, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt? x) -> SymInt")

        def foo_op_6(x: Sequence[int]) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_6, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        def foo_op_7(x: list[int]) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_7, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        def foo_op_8(x: Sequence[int] | None) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_8, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        def foo_op_9(x: list[int] | None) -> int:
            # [FUNCTION DOC] HELPER: ignores x and returns 1.
            return 1

        result = torch.library.infer_schema(foo_op_9, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        def foo_op_10(x: int | float | bool) -> types.Number:
            # [FUNCTION DOC] HELPER: returns x unchanged.
            return x

        result = torch.library.infer_schema(foo_op_10, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

        def foo_op_11(x: int | bool | float) -> types.Number:
            # [FUNCTION DOC] HELPER: returns x unchanged.
            return x

        result = torch.library.infer_schema(foo_op_11, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

    def test_unsupported_annotation(self):
        # [FUNCTION DOC] Annotations naming undefined types (a parameter D and a return E) must
        # make schema inference raise ValueError saying the annotation is not a type.
        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported type annotation D. It is not a type.",
        ):

            def foo_op(x: D) -> Tensor:  # noqa: F821
                # [FUNCTION DOC] HELPER: builds a tensor from x; never executed.
                return torch.Tensor(x)

            torch.library.infer_schema(foo_op, mutates_args=mutates_args)

        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported type annotation E. It is not a type.",
        ):

            def foo_op_2(x: Tensor) -> E:  # noqa: F821
                # [FUNCTION DOC] HELPER: returns x unchanged; never executed.
                return x

            torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)

    def test_name_error_hint(self):
        # [FUNCTION DOC] With postponed annotations and an unresolvable name, the ValueError must
        # mention the postponed-annotations feature as a hint about module scope.
        # When from __future__ import annotations is active and a type name cannot
        # be resolved (e.g. because it was only imported inside a nested function),
        # the error should include the original NameError and a hint about module scope.
        with self.assertRaisesRegex(
            ValueError,
            r"from __future__ import annotations",
        ):

            def foo_op(x: D) -> Tensor:  # noqa: F821
                # [FUNCTION DOC] HELPER: builds a tensor from x; never executed.
                return torch.Tensor(x)

            torch.library.infer_schema(foo_op, mutates_args=mutates_args)


if __name__ == "__main__":
    run_tests()
