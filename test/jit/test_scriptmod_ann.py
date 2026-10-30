# Owner(s): ["oncall: jit"]
# [MODULE DOC] JIT tests that instance attributes of a scripted nn.Module keep their declared
# types (annotated falsy base types, non-empty containers, tensors, jit.Attribute, class-level
# annotations) and that empty list/dict/optional annotations are rejected with a warning plus an
# error. Defines one context-manager helper and one test class. Importing appends the test
# directory to sys.path; direct execution redirects to test/test_jit.py.

import contextlib
import os
import sys
import warnings
from typing import Dict, List, Optional

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


@contextlib.contextmanager
# [FUNCTION DOC] HELPER: Context manager that records warnings while ignoring FutureWarning
# messages that tell users to switch away from torch.jit.script, and yields the recorded list so
# tests can assert no other warnings were emitted. Warning filters are restored on exit.
def _catch_warnings_ignoring_jit_deprecation():
    # torch.jit.script emits a visible FutureWarning; ignore it so these tests
    # can assert on the absence of *unexpected* warnings.
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(
            "ignore", ".*Please switch to `torch.", category=FutureWarning
        )
        yield w


class TestScriptModuleInstanceAttributeTypeAnnotation(JitTestCase):
    # [CLASS DOC] Tests for type inference of annotated instance attributes on scripted modules.
    # Positive cases script a module via checkModule and expect zero warnings; empty container or
    # Optional cases expect a UserWarning about unsupported annotations plus a RuntimeError when
    # forward assigns to the attribute.
    # NB: There are no tests for `Tuple` or `NamedTuple` here. In fact,
    # reassigning a non-empty Tuple to an attribute previously typed
    # as containing an empty Tuple SHOULD fail. See note in `_check.py`

    def test_annotated_falsy_base_type(self):
        # [FUNCTION DOC] Int attribute annotated and set to zero in init, overwritten in forward;
        # asserts checkModule succeeds and no warnings are recorded.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an int attribute
            # (zero in init) that forward overwrites.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets an annotated int attribute to zero.
                super().__init__()
                self.x: int = 0

            def forward(self, x: int):
                # [FUNCTION DOC] HELPER: Stores the int argument in the attribute and returns one.
                self.x = x
                return 1

        with _catch_warnings_ignoring_jit_deprecation() as w:
            self.checkModule(M(), (1,))
        if len(w) != 0:
            raise AssertionError(f"Expected no warnings, got {len(w)}: {w}")

    def test_annotated_nonempty_container(self):
        # [FUNCTION DOC] Non-empty list attribute annotated in init and reassigned in forward;
        # asserts checkModule succeeds and no warnings are recorded.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a list-of-int attribute initialized to three items.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets an annotated list-of-int attribute to three items.
                super().__init__()
                self.x: List[int] = [1, 2, 3]

            def forward(self, x: List[int]):
                # [FUNCTION DOC] HELPER: Stores the list argument
                # in the attribute and returns one.
                self.x = x
                return 1

        with _catch_warnings_ignoring_jit_deprecation() as w:
            self.checkModule(M(), ([1, 2, 3],))
        if len(w) != 0:
            raise AssertionError(f"Expected no warnings, got {len(w)}: {w}")

    def test_annotated_empty_tensor(self):
        # [FUNCTION DOC] Tensor attribute annotated and initialized empty, reassigned in forward;
        # asserts checkModule succeeds and no warnings are recorded.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a tensor attribute initialized to an empty tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets an annotated tensor attribute to an empty tensor.
                super().__init__()
                self.x: torch.Tensor = torch.empty(0)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Stores the tensor argument
                # in the attribute and returns it.
                self.x = x
                return self.x

        with _catch_warnings_ignoring_jit_deprecation() as w:
            self.checkModule(M(), (torch.rand(2, 3),))
        if len(w) != 0:
            raise AssertionError(f"Expected no warnings, got {len(w)}: {w}")

    def test_annotated_with_jit_attribute(self):
        # [FUNCTION DOC] Attribute created via jit.Attribute with an empty list and list type;
        # asserts checkModule succeeds and no warnings are recorded.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose attribute is a
            # jit.Attribute holding an empty int list.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to an empty list typed as list of int.
                super().__init__()
                self.x = torch.jit.Attribute([], List[int])

            def forward(self, x: List[int]):
                # [FUNCTION DOC] HELPER: Stores the list argument in the attribute and returns it.
                self.x = x
                return self.x

        with _catch_warnings_ignoring_jit_deprecation() as w:
            self.checkModule(M(), ([1, 2, 3],))
        if len(w) != 0:
            raise AssertionError(f"Expected no warnings, got {len(w)}: {w}")

    def test_annotated_class_level_annotation_only(self):
        # [FUNCTION DOC] Attribute annotated only at class level (empty list in init); asserts
        # checkModule succeeds and no warnings are recorded.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a class-level list-of-int annotation and no
            # annotation in init.
            x: List[int]

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to an empty list.
                super().__init__()
                self.x = []

            def forward(self, y: List[int]):
                # [FUNCTION DOC] HELPER: Stores the list argument in the attribute and returns it.
                self.x = y
                return self.x

        with _catch_warnings_ignoring_jit_deprecation() as w:
            self.checkModule(M(), ([1, 2, 3],))
        if len(w) != 0:
            raise AssertionError(f"Expected no warnings, got {len(w)}: {w}")

    def test_annotated_class_level_annotation_and_init_annotation(self):
        # [FUNCTION DOC] Attribute annotated both at class level and in init (empty list); asserts
        # checkModule succeeds and no warnings are recorded.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with the same list-of-int
            # annotation on the class and in init.
            x: List[int]

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the annotated attribute to an empty list.
                super().__init__()
                self.x: List[int] = []

            def forward(self, y: List[int]):
                # [FUNCTION DOC] HELPER: Stores the list argument in the attribute and returns it.
                self.x = y
                return self.x

        with _catch_warnings_ignoring_jit_deprecation() as w:
            self.checkModule(M(), ([1, 2, 3],))
        if len(w) != 0:
            raise AssertionError(f"Expected no warnings, got {len(w)}: {w}")

    def test_annotated_class_level_jit_annotation(self):
        # [FUNCTION DOC] Class-level annotation plus an annotate call giving an empty int list in
        # init; asserts checkModule succeeds and no warnings are recorded.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a class-level annotation and an empty list annotated
            # in init.
            x: List[int]

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to an
                # empty list annotated as list of int.
                super().__init__()
                self.x: List[int] = torch.jit.annotate(List[int], [])

            def forward(self, y: List[int]):
                # [FUNCTION DOC] HELPER: Stores the list argument in the attribute and returns it.
                self.x = y
                return self.x

        with _catch_warnings_ignoring_jit_deprecation() as w:
            self.checkModule(M(), ([1, 2, 3],))
        if len(w) != 0:
            raise AssertionError(f"Expected no warnings, got {len(w)}: {w}")

    def test_annotated_empty_list(self):
        # [FUNCTION DOC] Asserts scripting yields a UserWarning about unsupported instance-level
        # annotations on empty non-base types, then a RuntimeError reporting a nonexistent
        # attribute being set, highlighting the assignment in forward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an empty list-of-int attribute annotated in init.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to an empty list.
                super().__init__()
                self.x: List[int] = []

            # [FUNCTION DOC] HELPER: Assigns the argument to the attribute and returns one.
            # KNOWN ISSUE: placed above the def because the test asserts highlighted source text
            # of the scripted forward body.
            def forward(self, x: List[int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_empty_list_lowercase(self):
        # [FUNCTION DOC] Asserts scripting yields a UserWarning about unsupported instance-level
        # annotations on empty non-base types, then a RuntimeError reporting a nonexistent
        # attribute being set, highlighting the assignment in forward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an empty built-in lowercase list-of-int attribute
            # annotated in init.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to an empty list.
                super().__init__()
                self.x: list[int] = []

            # [FUNCTION DOC] HELPER: Assigns the argument to the attribute and returns one.
            # KNOWN ISSUE: placed above the def because the test asserts highlighted source text
            # of the scripted forward body.
            def forward(self, x: list[int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_empty_dict(self):
        # [FUNCTION DOC] Asserts scripting yields a UserWarning about unsupported instance-level
        # annotations on empty non-base types, then a RuntimeError reporting a nonexistent
        # attribute being set, highlighting the assignment in forward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an empty
            # string-to-int dict attribute annotated in init.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to an empty dict.
                super().__init__()
                self.x: Dict[str, int] = {}

            # [FUNCTION DOC] HELPER: Assigns the argument to the attribute and returns one.
            # KNOWN ISSUE: placed above the def because the test asserts highlighted source text
            # of the scripted forward body.
            def forward(self, x: Dict[str, int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_empty_dict_lowercase(self):
        # [FUNCTION DOC] Asserts scripting yields a UserWarning about unsupported instance-level
        # annotations on empty non-base types, then a RuntimeError reporting a nonexistent
        # attribute being set, highlighting the assignment in forward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an empty lowercase dict attribute annotated in init.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to an empty dict.
                super().__init__()
                self.x: dict[str, int] = {}

            # [FUNCTION DOC] HELPER: Assigns the argument to the attribute and returns one.
            # KNOWN ISSUE: placed above the def because the test asserts highlighted source text
            # of the scripted forward body.
            def forward(self, x: dict[str, int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_empty_optional(self):
        # [FUNCTION DOC] Asserts scripting yields a UserWarning about unsupported instance-level
        # annotations on empty non-base types, then a RuntimeError reporting a wrong type for the
        # attribute assignment, highlighting the assignment in forward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an Optional string attribute annotated in init and
            # set to none.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to none.
                super().__init__()
                self.x: Optional[str] = None

            # [FUNCTION DOC] HELPER: Assigns the argument to the attribute and returns one.
            # KNOWN ISSUE: placed above the def because the test asserts highlighted source text
            # of the scripted forward body.
            def forward(self, x: Optional[str]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_list(self):
        # [FUNCTION DOC] Asserts scripting yields a UserWarning about unsupported instance-level
        # annotations on empty non-base types, then a RuntimeError reporting a nonexistent
        # attribute being set, highlighting the assignment in forward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose empty list
            # attribute is created with an annotate call.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to an
                # empty list annotated as list of int.
                super().__init__()
                self.x = torch.jit.annotate(List[int], [])

            # [FUNCTION DOC] HELPER: Assigns the argument to the attribute and returns one.
            # KNOWN ISSUE: placed above the def because the test asserts highlighted source text
            # of the scripted forward body.
            def forward(self, x: List[int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_list_lowercase(self):
        # [FUNCTION DOC] Asserts scripting yields a UserWarning about unsupported instance-level
        # annotations on empty non-base types, then a RuntimeError reporting a nonexistent
        # attribute being set, highlighting the assignment in forward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose empty list attribute is annotated with the
            # lowercase list type.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to an empty lowercase-typed list.
                super().__init__()
                self.x = torch.jit.annotate(list[int], [])

            # [FUNCTION DOC] HELPER: Assigns the argument to the attribute and returns one.
            # KNOWN ISSUE: placed above the def because the test asserts highlighted source text
            # of the scripted forward body.
            def forward(self, x: list[int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_dict(self):
        # [FUNCTION DOC] Asserts scripting yields a UserWarning about unsupported instance-level
        # annotations on empty non-base types, then a RuntimeError reporting a nonexistent
        # attribute being set, highlighting the assignment in forward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose empty dict
            # attribute is created with an annotate call.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to an empty annotated dict.
                super().__init__()
                self.x = torch.jit.annotate(Dict[str, int], {})

            # [FUNCTION DOC] HELPER: Assigns the argument to the attribute and returns one.
            # KNOWN ISSUE: placed above the def because the test asserts highlighted source text
            # of the scripted forward body.
            def forward(self, x: Dict[str, int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_dict_lowercase(self):
        # [FUNCTION DOC] Asserts scripting yields a UserWarning about unsupported instance-level
        # annotations on empty non-base types, then a RuntimeError reporting a nonexistent
        # attribute being set, highlighting the assignment in forward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose empty dict attribute is annotated with the
            # lowercase dict type.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to an empty lowercase-typed dict.
                super().__init__()
                self.x = torch.jit.annotate(dict[str, int], {})

            # [FUNCTION DOC] HELPER: Assigns the argument to the attribute and returns one.
            # KNOWN ISSUE: placed above the def because the test asserts highlighted source text
            # of the scripted forward body.
            def forward(self, x: dict[str, int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_optional(self):
        # [FUNCTION DOC] Asserts scripting yields a UserWarning about unsupported instance-level
        # annotations on empty non-base types, then a RuntimeError reporting a wrong type for the
        # attribute assignment, highlighting the assignment in forward.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose Optional string
            # attribute is annotated and set to none.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to none via an annotate call.
                super().__init__()
                self.x = torch.jit.annotate(Optional[str], None)

            # [FUNCTION DOC] HELPER: Assigns the argument to the attribute and returns one.
            # KNOWN ISSUE: placed above the def because the test asserts highlighted source text
            # of the scripted forward body.
            def forward(self, x: Optional[str]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_torch_jit_import(self):
        # [FUNCTION DOC] Asserts scripting yields a UserWarning about unsupported instance-level
        # annotations on empty non-base types, then a RuntimeError reporting a wrong type for the
        # attribute assignment, highlighting the assignment in forward.
        from torch import jit

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose Optional attribute uses the imported jit alias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to none via the jit alias annotate.
                super().__init__()
                self.x = jit.annotate(Optional[str], None)

            # [FUNCTION DOC] HELPER: Assigns the argument to the attribute and returns one.
            # KNOWN ISSUE: placed above the def because the test asserts highlighted source text
            # of the scripted forward body.
            def forward(self, x: Optional[str]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
