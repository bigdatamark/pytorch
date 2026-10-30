# Owner(s): ["oncall: mobile"]
# [MODULE DOC] Lite interpreter tests for TorchScript modules that use named tuple types
# (typing NamedTuple and collections namedtuple), including nested and custom class types.
# Each test scripts a module, saves it in lite interpreter format to a buffer, reloads it and
# compares the mobile result to the scripted result. No import-time side effects.

import io
import unittest
from collections import namedtuple
from typing import NamedTuple

import torch
import torch.utils.bundled_inputs
from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing._internal.common_utils import run_tests, TestCase


class TestLiteScriptModule(TestCase):
    # [CLASS DOC] TestCase of script versus mobile round-trip tests for named tuple types.
    def test_typing_namedtuple(self):
        # [FUNCTION DOC] Scripts a module whose forward wraps the input in a typing named
        # tuple holding a list of tensors, saves it to a lite buffer with debug info, reloads it
        # and checks the field a of the mobile result matches the scripted result.
        myNamedTuple = NamedTuple(  # noqa: UP014
            "myNamedTuple", [("a", list[torch.Tensor])]
        )

        class MyTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns a named tuple built from a one
            # element list holding the input tensor.
            def forward(self, a: torch.Tensor):
                # [FUNCTION DOC] HELPER: Builds and returns the named tuple wrapping the input.
                p = myNamedTuple([a])
                return p

        sample_input = torch.tensor(5)
        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(sample_input).a

        buffer = io.BytesIO(
            script_module._save_to_buffer_for_lite_interpreter(
                _save_mobile_debug_info=True
            )
        )
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)  # Error here
        mobile_module_result = mobile_module(sample_input).a
        torch.testing.assert_close(script_module_result, mobile_module_result)

    @unittest.skip("T137512434")
    def test_typing_dict_with_namedtuple(self):
        # [FUNCTION DOC] Skipped (ticket T137512434). Would script a module that stores a
        # NamedTuple attribute and builds a dict of str to NamedTuple, return both the
        # attribute and a dict entry, and compare scripted and mobile results after a lite
        # round trip.
        class Foo(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with a single tensor field id.
            id: torch.Tensor

        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a NamedTuple attribute foo that forward overwrites.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the foo attribute with the NamedTuple
                # holding a scalar tensor one.
                super().__init__()
                self.foo = Foo(torch.tensor(1))

            def forward(self, a: torch.Tensor):
                # [FUNCTION DOC] HELPER: Replaces foo with a NamedTuple of the input, stores
                # another one in a local dict under the key test, and returns the pair of
                # the attribute and the dict entry.
                self.foo = Foo(a)
                re: dict[str, Foo] = {}
                re["test"] = Foo(a)
                return self.foo, re["test"]

        # The corresponding bytecode is
        # (8,
        #  ('__torch__.___torch_mangle_2.Bar.forward',
        #   (('instructions',
        #     (('STOREN', 1, 2),
        #      ('DROPR', 1, 0),
        #      ('DICT_CONSTRUCT', 0, 0),
        #      ('STORE', 3, 0),
        #      ('LOAD', 3, 0),
        #      ('LOADC', 1, 0),
        #      ('MOVE', 2, 0),
        #      ('NAMED_TUPLE_CONSTRUCT', 1, 1),
        #      ('OP', 0, 0),
        #      ('MOVE', 3, 0),
        #      ('LOADC', 1, 0),
        #      ('DICT_INDEX', 0, 0),
        #      ('LOADC', 0, 0),
        #      ('TUPLE_INDEX', 0, 0),
        #      ('RET', 0, 0))),
        #    ('operators', (('aten::_set_item', 'str', 3),)),
        #    ('constants', (0, 'test')),
        #    ('types',
        #     ('Dict[str,__torch__.Foo[NamedTuple, [[id, Tensor]]]]',
        #      '__torch__.Foo[NamedTuple, [[id, Tensor]]]')),
        #    ('register_size', 3)),
        #   (('arguments',
        #     ((('name', 'self'),
        #       ('type', '__torch__.___torch_mangle_2.Bar'),
        #       ('default_value', None)),
        #      (('name', 'a'), ('type', 'Tensor'), ('default_value', None)))),
        #    ('returns',
        #     ((('name', ''), ('type', 'Tensor'), ('default_value', None)),)))))

        sample_input = torch.tensor(5)
        script_module = torch.jit.script(Bar())

        script_module_result = script_module(sample_input)

        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        mobile_module_result = mobile_module(sample_input)
        torch.testing.assert_close(script_module_result, mobile_module_result)

    def test_typing_namedtuple_custom_classtype(self):
        # [FUNCTION DOC] Scripts a module that rebuilds a custom NamedTuple attribute from the
        # input and returns it, round trips it through the lite interpreter and asserts the
        # mobile result equals the scripted result.
        class Foo(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with a single tensor field id.
            id: torch.Tensor

        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a NamedTuple attribute foo reassigned in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes foo with a NamedTuple of a scalar tensor.
                super().__init__()
                self.foo = Foo(torch.tensor(1))

            def forward(self, a: torch.Tensor):
                # [FUNCTION DOC] HELPER: Replaces foo with a NamedTuple built from the input
                # and returns it.
                self.foo = Foo(a)
                return self.foo

        sample_input = torch.tensor(5)
        script_module = torch.jit.script(Bar())
        script_module_result = script_module(sample_input)

        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        mobile_module_result = mobile_module(sample_input)
        torch.testing.assert_close(script_module_result, mobile_module_result)

    def test_return_collections_namedtuple(self):
        # [FUNCTION DOC] Scripts a module returning a collections namedtuple built from the
        # input, round trips it through the lite interpreter and asserts the mobile result
        # equals the scripted result.
        myNamedTuple = namedtuple("myNamedTuple", [("a")])

        class MyTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns a collections namedtuple.
            def forward(self, a: torch.Tensor):
                # [FUNCTION DOC] HELPER: Wraps the input tensor in the namedtuple and returns it.
                return myNamedTuple(a)

        sample_input = torch.Tensor(1)
        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(sample_input)
        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        mobile_module_result = mobile_module(sample_input)
        torch.testing.assert_close(script_module_result, mobile_module_result)

    def test_nest_typing_namedtuple_custom_classtype(self):
        # [FUNCTION DOC] Same round trip check as the custom class test but with a nested
        # NamedTuple: the outer tuple holds a tensor and an inner NamedTuple. Compares the
        # innermost field of the mobile and scripted results.
        class Baz(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with a single tensor field di (the inner type).
            di: torch.Tensor

        class Foo(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with a tensor field id and a Baz field baz.
            id: torch.Tensor
            baz: Baz

        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a nested NamedTuple attribute foo reset in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes foo with a scalar tensor and a Baz.
                super().__init__()
                self.foo = Foo(torch.tensor(1), Baz(torch.tensor(1)))

            def forward(self, a: torch.Tensor):
                # [FUNCTION DOC] HELPER: Replaces foo with a Foo built from the input and a new
                # Baz holding a scalar tensor, and returns it.
                self.foo = Foo(a, Baz(torch.tensor(1)))
                return self.foo

        sample_input = torch.tensor(5)
        script_module = torch.jit.script(Bar())
        script_module_result = script_module(sample_input)

        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        mobile_module_result = mobile_module(sample_input)
        torch.testing.assert_close(
            script_module_result.baz.di, mobile_module_result.baz.di
        )


if __name__ == "__main__":
    run_tests()
