# Owner(s): ["module: functorch"]
# [MODULE DOC] Tests for the generated subclass wrapper source produced by
# _codegen_subclass_wrapper_source in torch._functorch._aot_autograd.subclass_codegen. Covers end-
# to-end torch.compile on TwoTensor inputs (checking the logged wrapper source against an expected
# text), wait insertion for async collective tensor paths, forwarding of trailing arguments, and
# output slot bookkeeping when subclass symbolic size values are elided or mixed with plain tensor
# outputs. Defines a stand-in subclass metadata dataclass and a module logger for the structured
# trace log; no other import-time side effects.

import logging
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch._functorch.config
from torch._functorch._aot_autograd.schemas import PlainTensorMeta
from torch._functorch._aot_autograd.subclass_codegen import (
    _codegen_subclass_wrapper_source,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.two_tensor import TwoTensor


trace_log = logging.getLogger("torch.__trace")


@dataclass
class _TestSubclassMeta:
    # [CLASS DOC] HELPER: Lightweight stand-in for the subclass creation metadata, avoiding the
    # fake tensor check of the real class. Holds the flat tensor start index, argument count,
    # whether subclass symbolic sizes are included, a mapping of attribute names to inner
    # metadata, outer size and stride, optional attribute names the outer size or stride is
    # derived from, the meta object, the original subclass and its type.
    flat_tensor_start_idx: int
    arg_count: int
    included_subclass_symints: bool
    attrs: dict
    outer_size: tuple
    outer_stride: tuple
    meta: Any
    original_subclass: Any
    original_subclass_type: type
    outer_size_from_attr: str | None = None
    outer_stride_from_attr: str | None = None


class TestSubclassCodegen(TestCase):
    # [CLASS DOC] Tests of the generated subclass wrapper: expected generated text for
    # compile runs and code generation, and runtime behavior of the wrapper against mocked
    # compiled functions.
    def _two_tensor_meta(self) -> _TestSubclassMeta:
        # [FUNCTION DOC] HELPER: Build metadata for a TwoTensor with two plain inner tensors at
        # unwrapped positions zero and one, argument count two, outer size four and stride one,
        # and subclass symbolic sizes marked as included.
        return _TestSubclassMeta(
            flat_tensor_start_idx=0,
            arg_count=2,
            included_subclass_symints=True,
            attrs={
                "a": PlainTensorMeta(unwrapped_idx=0),
                "b": PlainTensorMeta(unwrapped_idx=1),
            },
            outer_size=(4,),
            outer_stride=(1,),
            meta=None,
            original_subclass=None,
            original_subclass_type=TwoTensor,
        )

    @contextmanager
    def _capture_wrapper_source(self):
        # [FUNCTION DOC] HELPER: Context manager that yields a list collecting payloads of
        # structured trace artifacts named subclass_wrapper. Installs a logging handler on the
        # trace logger and lowers its level to debug for the duration, then removes the handler
        # and restores the previous level, even when the body raises.
        """Capture subclass_wrapper artifacts from the structured trace log."""
        captured: list[str] = []

        class _ArtifactHandler(logging.Handler):
            # [CLASS DOC] HELPER: Logging handler that appends the payload of each record whose
            # artifact metadata is named subclass_wrapper to the enclosing captured list.
            def emit(self, record):
                # [FUNCTION DOC] HELPER: Handle one log record: read its metadata, and if it
                # carries the subclass_wrapper artifact and a non-None payload, append the payload
                # to the captured list.
                metadata = getattr(record, "metadata", {})
                if (
                    "artifact" in metadata
                    and metadata["artifact"].get("name") == "subclass_wrapper"
                ):
                    payload = getattr(record, "payload", None)
                    if payload is not None:
                        captured.append(payload)

        handler = _ArtifactHandler()
        handler.setLevel(logging.DEBUG)
        old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)
        trace_log.addHandler(handler)
        try:
            yield captured
        finally:
            trace_log.removeHandler(handler)
            trace_log.setLevel(old_level)

    def test_compile_simple(self):
        # [FUNCTION DOC] Compiles a doubling function with the eager AOT backend on a TwoTensor,
        # checks the output is a TwoTensor with doubled inner tensors, exactly one wrapper source
        # was logged, and that it matches the expected generated text.
        """torch.compile with TwoTensor produces correct wrapper and output."""
        with self._capture_wrapper_source() as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Function being compiled; returns its input multiplied
                # by two.
                return x * 2

            a = torch.randn(4)
            b = torch.randn(4)
            tt = TwoTensor(a, b)
            out = f(tt)

        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, a * 2)
        self.assertEqual(out.b, b * 2)

        self.assertEqual(len(captured), 1)
        self.assertExpectedInline(
            captured[0],
            """\
def inner_fn(args):
    unwrapped_args = []
    _inp_0 = args[0]
    assert type(_inp_0) is _expected_type_1, f'expected {_expected_type_1}, got {type(_inp_0)}'
    unwrapped_args.append(_inp_0.a)
    unwrapped_args.append(_inp_0.b)
    unwrapped_args.extend(args[1:])
    args.clear()
    unwrapped_outs = compiled_fn(unwrapped_args)
    _out_idx = 0
    _num_wrapped_outs = len(unwrapped_outs)
    assert _num_wrapped_outs == 2, f'expected 2 wrapped outputs, got {_num_wrapped_outs}'
    _out_attr_3 = unwrapped_outs[_out_idx]
    _out_idx += 1
    _out_attr_4 = unwrapped_outs[_out_idx]
    _out_idx += 1
    _out_inner_2 = {'a': _out_attr_3, 'b': _out_attr_4}
    _out_7 = _subclass_type_5.__tensor_unflatten__(_out_inner_2, _meta_6, _out_attr_3.size(), _out_attr_3.stride())
    assert _out_idx == _num_wrapped_outs, f'wrapped {_out_idx} outputs, expected {_num_wrapped_outs}'
    return (_out_7,)""",
        )

    def test_compile_nested_subclass(self):
        # [FUNCTION DOC] Compiles a sine-then-cosine function on a TwoTensor whose two inner
        # tensors are themselves TwoTensors; checks nested outputs and values, one logged wrapper,
        # and the expected recursive generated text.
        """torch.compile with nested TwoTensor produces correct recursive wrapper."""
        with self._capture_wrapper_source() as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Function being compiled; applies sine then cosine to
                # its input.
                return x.sin().cos()

            a1 = torch.randn(4)
            a2 = torch.randn(4)
            a3 = torch.randn(4)
            a4 = torch.randn(4)
            inner_a = TwoTensor(a1, a2)
            inner_b = TwoTensor(a3, a4)
            tt = TwoTensor(inner_a, inner_b)
            out = f(tt)

        self.assertIsInstance(out, TwoTensor)
        self.assertIsInstance(out.a, TwoTensor)
        self.assertIsInstance(out.b, TwoTensor)
        self.assertEqual(out.a.a, inner_a.a.sin().cos())
        self.assertEqual(out.b.b, inner_b.b.sin().cos())

        self.assertEqual(len(captured), 1)
        self.assertExpectedInline(
            captured[0],
            """\
def inner_fn(args):
    unwrapped_args = []
    _inp_0 = args[0]
    assert type(_inp_0) is _expected_type_1, f'expected {_expected_type_1}, got {type(_inp_0)}'
    _inner_2 = _inp_0.a
    unwrapped_args.append(_inner_2.a)
    unwrapped_args.append(_inner_2.b)
    _inner_3 = _inp_0.b
    unwrapped_args.append(_inner_3.a)
    unwrapped_args.append(_inner_3.b)
    unwrapped_args.extend(args[1:])
    args.clear()
    unwrapped_outs = compiled_fn(unwrapped_args)
    _out_idx = 0
    _num_wrapped_outs = len(unwrapped_outs)
    assert _num_wrapped_outs == 4, f'expected 4 wrapped outputs, got {_num_wrapped_outs}'
    _out_attr_6 = unwrapped_outs[_out_idx]
    _out_idx += 1
    _out_attr_7 = unwrapped_outs[_out_idx]
    _out_idx += 1
    _out_inner_5 = {'a': _out_attr_6, 'b': _out_attr_7}
    _out_10 = _subclass_type_8.__tensor_unflatten__(_out_inner_5, _meta_9, _out_attr_6.size(), _out_attr_6.stride())
    _out_attr_12 = unwrapped_outs[_out_idx]
    _out_idx += 1
    _out_attr_13 = unwrapped_outs[_out_idx]
    _out_idx += 1
    _out_inner_11 = {'a': _out_attr_12, 'b': _out_attr_13}
    _out_16 = _subclass_type_14.__tensor_unflatten__(_out_inner_11, _meta_15, _out_attr_12.size(), _out_attr_12.stride())
    _out_inner_4 = {'a': _out_10, 'b': _out_16}
    _out_19 = _subclass_type_17.__tensor_unflatten__(_out_inner_4, _meta_18, _out_10.size(), _out_10.stride())
    assert _out_idx == _num_wrapped_outs, f'wrapped {_out_idx} outputs, expected {_num_wrapped_outs}'
    return (_out_19,)""",
        )

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_act_wait_top_level_codegen(self):
        # [FUNCTION DOC] Skipped without distributed support. Generates wrapper source for plain
        # tensor inputs with an async collective tensor path at the top level and checks the text
        # waits on the first argument while unwrapping.
        """ACT paths on plain inputs emit waits during input unwrapping."""
        source, _ = _codegen_subclass_wrapper_source(
            inp_metas=[PlainTensorMeta(unwrapped_idx=0)],
            out_metas=[PlainTensorMeta(unwrapped_idx=0)],
            num_fw_outs_saved_for_bw=None,
            act_input_paths=[(0, ())],
        )

        self.assertExpectedInline(
            source,
            """\
def inner_fn(args):
    unwrapped_args = []
    unwrapped_args.append(_maybe_wait_act_0(args[0]))
    unwrapped_args.extend(args[1:])
    args.clear()
    unwrapped_outs = compiled_fn(unwrapped_args)
    _out_idx = 0
    _num_wrapped_outs = len(unwrapped_outs)
    assert _num_wrapped_outs == 1, f'expected 1 wrapped outputs, got {_num_wrapped_outs}'
    _out_plain_1 = unwrapped_outs[_out_idx]
    _out_idx += 1
    assert _out_idx == _num_wrapped_outs, f'wrapped {_out_idx} outputs, expected {_num_wrapped_outs}'
    return (_out_plain_1,)""",
        )

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_act_wait_nested_codegen(self):
        # [FUNCTION DOC] Skipped without distributed support. Generates wrapper source for a
        # TwoTensor input with an async collective path naming inner attribute a, and checks the
        # wait is applied to that leaf only, via the expected text.
        """ACT paths inside subclasses emit waits at the matching leaf."""
        source, _ = _codegen_subclass_wrapper_source(
            inp_metas=[self._two_tensor_meta()],
            out_metas=[PlainTensorMeta(unwrapped_idx=0)],
            num_fw_outs_saved_for_bw=None,
            act_input_paths=[(0, ("a",))],
        )

        self.assertExpectedInline(
            source,
            """\
def inner_fn(args):
    unwrapped_args = []
    _inp_1 = args[0]
    assert type(_inp_1) is _expected_type_2, f'expected {_expected_type_2}, got {type(_inp_1)}'
    _resolved_3 = _maybe_wait_act_0(_inp_1.a)
    unwrapped_args.append(_resolved_3)
    unwrapped_args.append(_inp_1.b)
    unwrapped_args.extend(args[1:])
    args.clear()
    unwrapped_outs = compiled_fn(unwrapped_args)
    _out_idx = 0
    _num_wrapped_outs = len(unwrapped_outs)
    assert _num_wrapped_outs == 1, f'expected 1 wrapped outputs, got {_num_wrapped_outs}'
    _out_plain_4 = unwrapped_outs[_out_idx]
    _out_idx += 1
    assert _out_idx == _num_wrapped_outs, f'wrapped {_out_idx} outputs, expected {_num_wrapped_outs}'
    return (_out_plain_4,)""",
        )

    def test_trailing_args_forwarded(self):
        # [FUNCTION DOC] Generates a wrapper for one TwoTensor input and output, runs it with
        # extra trailing arguments (a seed and an offset, as the RNG wrapper appends) and checks
        # they are passed through to the compiled function after the unwrapped inner tensors, with
        # identity preserved.
        """Extra trailing args (e.g. rng seed/offset) are forwarded to compiled_fn."""
        # Build SubclassCreationMeta manually to avoid __post_init__ fake tensor check
        inp_meta = self._two_tensor_meta()
        out_meta = self._two_tensor_meta()

        source, globals_dict = _codegen_subclass_wrapper_source(
            inp_metas=[inp_meta],
            out_metas=[out_meta],
            num_fw_outs_saved_for_bw=None,
        )

        received_args = []

        def mock_compiled_fn(args):
            # [FUNCTION DOC] HELPER: Mock compiled function: records received arguments and
            # returns the first two doubled.
            received_args.extend(args)
            return [args[0] * 2, args[1] * 2]

        globals_dict["compiled_fn"] = mock_compiled_fn
        local_dict = {}
        exec(compile(source, "<test>", "exec"), globals_dict, local_dict)
        wrapper = local_dict["inner_fn"]

        a = torch.randn(4)
        b = torch.randn(4)
        tt = TwoTensor(a, b)
        seed = torch.tensor(42)
        offset = torch.tensor(0)
        # Simulate FunctionalizedRngRuntimeWrapper appending rng state
        wrapper([tt, seed, offset])

        self.assertEqual(len(received_args), 4)
        self.assertEqual(received_args[0], a)
        self.assertEqual(received_args[1], b)
        self.assertIs(received_args[2], seed)
        self.assertIs(received_args[3], offset)

    def test_attr_derived_metadata_does_not_skip_outputs(self):
        # [FUNCTION DOC] Builds output metadata whose outer size and stride derive from attribute
        # a, with a saved activation, and checks the wrapper still reads all outputs correctly: a
        # TwoTensor built from the first two outputs and the trailing activation returned by
        # identity, even though symbolic size values follow the tensors.
        # Build SubclassCreationMeta manually to avoid __post_init__ fake tensor check
        out_meta = _TestSubclassMeta(
            flat_tensor_start_idx=0,
            arg_count=5,
            included_subclass_symints=True,
            attrs={
                "a": PlainTensorMeta(unwrapped_idx=0),
                "b": PlainTensorMeta(unwrapped_idx=1),
            },
            outer_size=(None, None),
            outer_stride=(None, 1),
            meta=None,
            original_subclass=None,
            original_subclass_type=TwoTensor,
            outer_size_from_attr="a",
            outer_stride_from_attr="a",
        )

        source, globals_dict = _codegen_subclass_wrapper_source(
            inp_metas=[],
            out_metas=[out_meta],
            num_fw_outs_saved_for_bw=1,
        )

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        activation = torch.randn(4)

        def mock_compiled_fn(args):
            # [FUNCTION DOC] HELPER: Mock compiled function: returns two tensors, three integer
            # size and stride values and an activation tensor.
            return [a, b, a.shape[0], a.shape[1], a.stride()[0], activation]

        globals_dict["compiled_fn"] = mock_compiled_fn
        local_dict = {}
        exec(compile(source, "<test>", "exec"), globals_dict, local_dict)
        wrapper = local_dict["inner_fn"]

        out, saved_activation = wrapper([])

        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, a)
        self.assertEqual(out.b, b)
        self.assertIs(saved_activation, activation)

    def test_attr_derived_metadata_consumes_symints_between_outputs(self):
        # [FUNCTION DOC] Two consecutive TwoTensor outputs with attribute-derived size and stride;
        # checks the wrapper skips the symbolic size values that sit between the outputs and
        # rebuilds both subclass outputs from the correct slots.
        # Build SubclassCreationMeta manually to avoid __post_init__ fake tensor check
        first_meta = _TestSubclassMeta(
            flat_tensor_start_idx=0,
            arg_count=5,
            included_subclass_symints=True,
            attrs={
                "a": PlainTensorMeta(unwrapped_idx=0),
                "b": PlainTensorMeta(unwrapped_idx=1),
            },
            outer_size=(None, None),
            outer_stride=(None, 1),
            meta=None,
            original_subclass=None,
            original_subclass_type=TwoTensor,
            outer_size_from_attr="a",
            outer_stride_from_attr="a",
        )
        second_meta = _TestSubclassMeta(
            flat_tensor_start_idx=5,
            arg_count=5,
            included_subclass_symints=True,
            attrs={
                "a": PlainTensorMeta(unwrapped_idx=5),
                "b": PlainTensorMeta(unwrapped_idx=6),
            },
            outer_size=(None, None),
            outer_stride=(None, 1),
            meta=None,
            original_subclass=None,
            original_subclass_type=TwoTensor,
            outer_size_from_attr="a",
            outer_stride_from_attr="a",
        )

        source, globals_dict = _codegen_subclass_wrapper_source(
            inp_metas=[],
            out_metas=[first_meta, second_meta],
            num_fw_outs_saved_for_bw=None,
        )

        a0 = torch.randn(2, 3)
        b0 = torch.randn(2, 3)
        a1 = torch.randn(4, 5)
        b1 = torch.randn(4, 5)

        def mock_compiled_fn(args):
            # [FUNCTION DOC] HELPER: Mock compiled function: returns both tensor pairs, each
            # followed by its size and stride integers.
            return [
                a0,
                b0,
                a0.shape[0],
                a0.shape[1],
                a0.stride()[0],
                a1,
                b1,
                a1.shape[0],
                a1.shape[1],
                a1.stride()[0],
            ]

        globals_dict["compiled_fn"] = mock_compiled_fn
        local_dict = {}
        exec(compile(source, "<test>", "exec"), globals_dict, local_dict)
        wrapper = local_dict["inner_fn"]

        first, second = wrapper([])

        self.assertIsInstance(first, TwoTensor)
        self.assertEqual(first.a, a0)
        self.assertEqual(first.b, b0)
        self.assertIsInstance(second, TwoTensor)
        self.assertEqual(second.a, a1)
        self.assertEqual(second.b, b1)

    def test_plain_output_after_elided_subclass_symints(self):
        # [FUNCTION DOC] A subclass output followed by a plain tensor output, where the
        # partitioner dropped the unused symbolic size values; checks the plain output is read
        # from the next sequential runtime slot rather than its larger metadata index.
        # Build SubclassCreationMeta manually to avoid __post_init__ fake tensor check
        subclass_meta = _TestSubclassMeta(
            flat_tensor_start_idx=0,
            arg_count=5,
            included_subclass_symints=True,
            attrs={
                "a": PlainTensorMeta(unwrapped_idx=0),
                "b": PlainTensorMeta(unwrapped_idx=1),
            },
            outer_size=(None, None),
            outer_stride=(None, 1),
            meta=None,
            original_subclass=None,
            original_subclass_type=TwoTensor,
            outer_size_from_attr="a",
            outer_stride_from_attr="a",
        )
        plain_meta = PlainTensorMeta(unwrapped_idx=5)

        source, globals_dict = _codegen_subclass_wrapper_source(
            inp_metas=[],
            out_metas=[subclass_meta, plain_meta],
            num_fw_outs_saved_for_bw=None,
        )

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        plain = torch.randn(4)

        def mock_compiled_fn(args):
            # [FUNCTION DOC] HELPER: Mock compiled function: returns the two inner tensors and the
            # plain tensor only.
            # The partitioner may drop optional subclass SymInt outputs when they
            # are unused. The following plain output must still be read from the
            # next sequential runtime output slot, not from its longer-layout
            # metadata index.
            return [a, b, plain]

        globals_dict["compiled_fn"] = mock_compiled_fn
        local_dict = {}
        exec(compile(source, "<test>", "exec"), globals_dict, local_dict)
        wrapper = local_dict["inner_fn"]

        subclass_out, plain_out = wrapper([])

        self.assertIsInstance(subclass_out, TwoTensor)
        self.assertEqual(subclass_out.a, a)
        self.assertEqual(subclass_out.b, b)
        self.assertIs(plain_out, plain)

    def test_plain_tensor_symint_outputs_can_be_mixed(self):
        # [FUNCTION DOC] Two plain tensor outputs with symbolic size and stride placeholders;
        # checks the wrapper returns the right tensors for two runtime layouts of the
        # compiled outputs, with the integers following the first tensor, then with them
        # following the second.
        first_meta = PlainTensorMeta(
            unwrapped_idx=0,
            size_symbol_placeholders=(True,),
            stride_symbol_placeholders=(True,),
        )
        second_meta = PlainTensorMeta(
            unwrapped_idx=3,
            size_symbol_placeholders=(True,),
            stride_symbol_placeholders=(True,),
        )

        source, globals_dict = _codegen_subclass_wrapper_source(
            inp_metas=[],
            out_metas=[first_meta, second_meta],
            num_fw_outs_saved_for_bw=None,
        )

        first = torch.randn(2)
        second = torch.randn(3)
        outputs = [first, first.shape[0], first.stride()[0], second]

        def mock_compiled_fn(args):
            # [FUNCTION DOC] HELPER: Mock compiled function: returns whichever outputs list is
            # currently bound in the enclosing test.
            return outputs

        globals_dict["compiled_fn"] = mock_compiled_fn
        local_dict = {}
        exec(compile(source, "<test>", "exec"), globals_dict, local_dict)
        wrapper = local_dict["inner_fn"]

        first_out, second_out = wrapper([])
        self.assertIs(first_out, first)
        self.assertIs(second_out, second)

        outputs = [first, second, second.shape[0], second.stride()[0]]
        first_out, second_out = wrapper([])
        self.assertIs(first_out, first)
        self.assertIs(second_out, second)

    def test_partial_elided_subclass_symints_between_outputs(self):
        # [FUNCTION DOC] A subclass output with attribute-derived size and stride followed by a
        # plain output, where only part of the symbolic size values is present; checks the wrapper
        # still returns the subclass output and the plain output by identity.
        # Build SubclassCreationMeta manually to avoid __post_init__ fake tensor check
        subclass_meta = _TestSubclassMeta(
            flat_tensor_start_idx=0,
            arg_count=5,
            included_subclass_symints=True,
            attrs={
                "a": PlainTensorMeta(unwrapped_idx=0),
                "b": PlainTensorMeta(unwrapped_idx=1),
            },
            outer_size=(None, None),
            outer_stride=(None, 1),
            meta=None,
            original_subclass=None,
            original_subclass_type=TwoTensor,
            outer_size_from_attr="a",
            outer_stride_from_attr="a",
        )
        plain_meta = PlainTensorMeta(unwrapped_idx=5)

        source, globals_dict = _codegen_subclass_wrapper_source(
            inp_metas=[],
            out_metas=[subclass_meta, plain_meta],
            num_fw_outs_saved_for_bw=None,
        )

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        plain = torch.randn(4)

        def mock_compiled_fn(args):
            # [FUNCTION DOC] HELPER: Mock compiled function: returns the two inner tensors, one
            # size integer and the plain tensor.
            return [a, b, a.shape[0], plain]

        globals_dict["compiled_fn"] = mock_compiled_fn
        local_dict = {}
        exec(compile(source, "<test>", "exec"), globals_dict, local_dict)
        wrapper = local_dict["inner_fn"]

        subclass_out, plain_out = wrapper([])

        self.assertIsInstance(subclass_out, TwoTensor)
        self.assertEqual(subclass_out.a, a)
        self.assertEqual(subclass_out.b, b)
        self.assertIs(plain_out, plain)

    def test_inner_plain_tensor_symints_forwarded(self):
        # [FUNCTION DOC] A TwoTensor whose inner tensors carry symbolic size and stride
        # placeholders; checks the unwrapped arguments passed to the compiled function contain the
        # first tensor, the size values four and six, then the second tensor, and the output is
        # rebuilt as a TwoTensor with doubled inner tensors.
        # Build SubclassCreationMeta manually to avoid __post_init__ fake tensor check
        inner_a_meta = PlainTensorMeta(
            unwrapped_idx=0,
            size_symbol_placeholders=(True, False),
            stride_symbol_placeholders=(True, False),
        )
        inner_b_meta = PlainTensorMeta(unwrapped_idx=3)
        inp_meta = _TestSubclassMeta(
            flat_tensor_start_idx=0,
            arg_count=4,
            included_subclass_symints=True,
            attrs={"a": inner_a_meta, "b": inner_b_meta},
            outer_size=(4, 6),
            outer_stride=(6, 1),
            meta=None,
            original_subclass=None,
            original_subclass_type=TwoTensor,
        )
        out_meta = _TestSubclassMeta(
            flat_tensor_start_idx=0,
            arg_count=4,
            included_subclass_symints=True,
            attrs={"a": inner_a_meta, "b": inner_b_meta},
            outer_size=(4, 6),
            outer_stride=(6, 1),
            meta=None,
            original_subclass=None,
            original_subclass_type=TwoTensor,
        )

        source, globals_dict = _codegen_subclass_wrapper_source(
            inp_metas=[inp_meta],
            out_metas=[out_meta],
            num_fw_outs_saved_for_bw=None,
        )

        received_args = []

        def mock_compiled_fn(args):
            # [FUNCTION DOC] HELPER: Mock compiled function: records received arguments and
            # returns doubled first and last tensors with the two integers passed through.
            received_args.extend(args)
            return [args[0] * 2, args[1], args[2], args[3] * 2]

        globals_dict["compiled_fn"] = mock_compiled_fn
        local_dict = {}
        exec(compile(source, "<test>", "exec"), globals_dict, local_dict)
        wrapper = local_dict["inner_fn"]

        a = torch.randn(4, 6)
        b = torch.randn(4, 6)
        out = wrapper([TwoTensor(a, b)])

        self.assertEqual(received_args[:4], [a, 4, 6, b])
        self.assertIsInstance(out[0], TwoTensor)
        self.assertEqual(out[0].a, a * 2)
        self.assertEqual(out[0].b, b * 2)


if __name__ == "__main__":
    run_tests()
