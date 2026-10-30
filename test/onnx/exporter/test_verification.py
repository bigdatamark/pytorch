# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests for the ONNX exporter verification module: the verification info record
# (creation from tensors and conversion to a mapping), the node-by-node verification
# interpreter, and the program-level verification function. Export tests use the dynamo
# exporter on tiny add/subtract modules.
"""Test the verification module."""

from __future__ import annotations

import json

import torch
from torch.onnx._internal.exporter import _verification
from torch.testing._internal import common_utils


class VerificationInfoTest(common_utils.TestCase):
    # [CLASS DOC] Tests for creating and serializing verification info records.
    def test_from_tensors(self):
        # [FUNCTION DOC] Builds a verification info from two identical float tensors and
        # checks the name, zero maximum absolute and relative differences, histogram counts
        # of three in the first bin, and float32 dtypes.
        # Test with tensors
        expected = torch.tensor([1.0, 2.0, 3.0])
        actual = torch.tensor([1.0, 2.0, 3.0])
        verification_info = _verification.VerificationInfo.from_tensors(
            "test_tensor", expected, actual
        )
        self.assertEqual(verification_info.name, "test_tensor")
        self.assertEqual(verification_info.max_abs_diff, 0)
        self.assertEqual(verification_info.max_rel_diff, 0)
        torch.testing.assert_close(
            verification_info.abs_diff_hist[0], torch.tensor([3.0] + [0.0] * 8)
        )
        torch.testing.assert_close(
            verification_info.rel_diff_hist[0], torch.tensor([3.0] + [0.0] * 8)
        )
        self.assertEqual(verification_info.expected_dtype, torch.float32)
        self.assertEqual(verification_info.actual_dtype, torch.float32)

    def test_from_tensors_int(self):
        # [FUNCTION DOC] Builds a verification info from a one-element integer tensor and
        # the plain integer one, checking the name, zero differences, histograms with a
        # single count in the first bin, and that both dtypes are 64-bit integers (the
        # scalar is converted to a tensor).
        # Test with int tensors
        expected = torch.tensor([1])
        actual = 1
        verification_info = _verification.VerificationInfo.from_tensors(
            "test_tensor_int", expected, actual
        )
        self.assertEqual(verification_info.name, "test_tensor_int")
        self.assertEqual(verification_info.max_abs_diff, 0)
        self.assertEqual(verification_info.max_rel_diff, 0)
        torch.testing.assert_close(
            verification_info.abs_diff_hist[0], torch.tensor([1.0] + [0.0] * 8)
        )
        torch.testing.assert_close(
            verification_info.rel_diff_hist[0], torch.tensor([1.0] + [0.0] * 8)
        )
        self.assertEqual(verification_info.expected_dtype, torch.int64)
        self.assertEqual(verification_info.actual_dtype, torch.int64)

    def test_asdict(self):
        # [FUNCTION DOC] Builds a verification info for identical float tensors and
        # checks the mapping form: scalar fields, histogram counts and bin edges as
        # nested lists, and dtypes as strings. Also checks the mapping survives a JSON
        # dump and load unchanged.
        # Test the asdict method
        expected = torch.tensor([1.0, 2.0, 3.0])
        actual = torch.tensor([1.0, 2.0, 3.0])
        verification_info = _verification.VerificationInfo.from_tensors(
            "test_tensor", expected, actual
        )
        asdict_result = verification_info.asdict()
        self.assertEqual(asdict_result["name"], "test_tensor")
        self.assertEqual(asdict_result["max_abs_diff"], 0)
        self.assertEqual(asdict_result["max_rel_diff"], 0)
        self.assertEqual(
            asdict_result["abs_diff_hist"],
            [
                [3.0] + [0.0] * 8,
                [0.0, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 1000000.0],
            ],
        )
        self.assertEqual(
            asdict_result["rel_diff_hist"],
            [
                [3.0] + [0.0] * 8,
                [0.0, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 1000000.0],
            ],
        )
        self.assertEqual(asdict_result["expected_dtype"], "torch.float32")
        self.assertEqual(asdict_result["actual_dtype"], "torch.float32")
        # Ensure it can be round tripped as json
        json_str = json.dumps(asdict_result)
        loaded_dict = json.loads(json_str)
        self.assertEqual(loaded_dict, asdict_result)


class VerificationInterpreterTest(common_utils.TestCase):
    # [CLASS DOC] Tests for the interpreter that evaluates an exported module while
    # comparing each node value with the ONNX result.
    def test_interpreter_stores_correct_info(self):
        # [FUNCTION DOC] Exports a small module through the dynamo path, runs the
        # verification interpreter on the sample arguments, and checks the returned result
        # matches eager execution, that three entries were recorded, and that every
        # entry has zero absolute and relative difference. Raises an assertion error if
        # export returns nothing.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module that adds its two inputs and returns the
            # sum minus one.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the sum of the two inputs minus one.
                c = a + b
                return c - 1

        model = Model()
        args = (torch.tensor([1.0]), torch.tensor([2.0]))
        onnx_program = torch.onnx.export(model, args, dynamo=True, verbose=False)
        if onnx_program is None:
            raise AssertionError("onnx_program is None")
        interpreter = _verification._VerificationInterpreter(onnx_program)
        results = interpreter.run(args)
        torch.testing.assert_close(results, model(*args))
        verification_infos = interpreter.verification_infos
        self.assertEqual(len(verification_infos), 3)
        for info in verification_infos:
            self.assertEqual(info.max_abs_diff, 0)
            self.assertEqual(info.max_rel_diff, 0)


class VerificationFunctionsTest(common_utils.TestCase):
    # [CLASS DOC] Tests for the module-level verification function that compares an ONNX
    # program against its exported program.
    def test_verify_onnx_program(self):
        # [FUNCTION DOC] Exports a small two-output module through the dynamo path, verifies
        # it with intermediate comparison disabled, and asserts that exactly two entries
        # come back, one per graph output.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose forward adds its two inputs and returns
            # a pair of the sum minus one and the sum.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Computes the sum of the two inputs and returns the
                # sum minus one together with the sum.
                c = a + b
                return c - 1, c

        model = Model()
        args = (torch.tensor([1.0]), torch.tensor([2.0]))
        onnx_program = torch.onnx.export(model, args, dynamo=True, verbose=False)
        if onnx_program is None:
            raise AssertionError("onnx_program is None")
        verification_infos = _verification.verify_onnx_program(
            onnx_program, args, compare_intermediates=False
        )
        self.assertEqual(len(verification_infos), 2)

    def test_verify_onnx_program_with_compare_intermediates_true(self):
    # [FUNCTION DOC] Exports a small two-output module through the dynamo path, verifies it
    # with intermediate comparison enabled, and asserts that three verification entries come
    # back (more than the two graph outputs).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose forward adds its two inputs and returns
            # a pair of the sum minus one and the sum.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Computes the sum of the two inputs and returns the
                # sum minus one together with the sum.
                c = a + b
                return c - 1, c

        model = Model()
        args = (torch.tensor([1.0]), torch.tensor([2.0]))
        onnx_program = torch.onnx.export(model, args, dynamo=True, verbose=False)
        if onnx_program is None:
            raise AssertionError("onnx_program is None")
        verification_infos = _verification.verify_onnx_program(
            onnx_program, args, compare_intermediates=True
        )
        self.assertEqual(len(verification_infos), 3)


if __name__ == "__main__":
    common_utils.run_tests()
