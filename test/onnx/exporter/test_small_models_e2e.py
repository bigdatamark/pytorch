# Owner(s): ["module: onnx"]
# [MODULE DOC] End-to-end tests of the ONNX dynamo exporter on small hand-written models.
# Defines a mixin that wraps the export call and one large parametrized test case covering
# control flow (conditionals, while loops), dtypes (complex, float8, float4, bfloat16),
# dynamic shapes, logging and print in forward, and many other export scenarios. Each test
# exports a model and usually checks numerical agreement with the testing helper. Importing
# requires onnx_ir, pytest and transformers.
"""Unit tests for the onnx dynamo exporter."""

from __future__ import annotations

import logging

import onnx_ir as ir
import pytest
import transformers

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.testing._internal import common_utils
from torch.utils import _pytree as torch_pytree


class _WithExport:
    # [CLASS DOC] HELPER: Mixin giving test cases an export method that always uses
    # the dynamo exporter.
    def export(self, model, args=(), kwargs=None, **options) -> torch.onnx.ONNXProgram:
        # [FUNCTION DOC] HELPER: Exports a model with the dynamo-based exporter and
        # non-verbose output. Puts module models in evaluation mode first (mutating them),
        # forwards args, keyword arguments and any extra export options, and raises an
        # assertion error if no program is returned. Returns the ONNX program.
        if isinstance(model, torch.nn.Module):
            model = model.eval()
        onnx_program = torch.onnx.export(
            model,
            args,
            kwargs=kwargs,
            dynamo=True,
            verbose=False,
            **options,
        )
        if onnx_program is None:
            raise AssertionError("Exported ONNX program is None")
        return onnx_program


@common_utils.instantiate_parametrized_tests
class DynamoExporterTest(common_utils.TestCase, _WithExport):
    # [CLASS DOC] Test case holding the dynamo exporter end-to-end tests; mixes in the
    # export helper and is expanded by the parametrization decorator.
    def test_insert_contiguous_between_transpose_and_view(self):
        # [FUNCTION DOC] Checks that exporting scaled dot product attention followed by
        # transpose and view works for half precision inputs: the exported program has no
        # method-call nodes, and the ONNX output matches within loose tolerances.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying attention to query, key and
            # value, transposing the first two dimensions and viewing the result
            # as three dimensions.
            def forward(self, query, key, value):
                # [FUNCTION DOC] HELPER: Computes scaled dot product attention, swaps
                # dimensions zero and one, and returns the result viewed with a
                # flattened last dimension.
                res = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value
                )
                rest = res.transpose(0, 1)
                return rest.view(8, 32, 128 * 64)

        model = Model()

        query = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16)

        ep = torch.export.export(model, (query, key, value), strict=False)
        self.assertNotIn("call_method", str(ep.graph))

        onnx_program = self.export(model, (query, key, value))
        onnx_testing.assert_onnx_program(onnx_program, atol=1e-3, rtol=1)

    def test_constant_complex(self):
        # [FUNCTION DOC] Exports a model multiplying a complex tensor by a complex Python
        # constant through the aten multiply operator and checks numerical agreement.
        class MulModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Model multiplying its input by a constant complex number.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input multiplied by the constant two
                # plus three j via the aten multiply operator.
                y = 2 + 3j
                return torch.ops.aten.mul(x, y)

        # Example usage with complex inputs
        x = torch.tensor(
            [[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]], dtype=torch.complex64
        )

        onnx_program = self.export(MulModule(), (x,))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_pow_does_not_trigger_type_promotion(self):
        # [FUNCTION DOC] Exports a half precision power by a float exponent and checks
        # numerical agreement and that the graph contains no cast node, meaning no type
        # promotion was inserted.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model raising its input to a floating point power.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input raised to the power two
                # as a float.
                return x**2.0

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)

        onnx_program = self.export(Model(), (x,))
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertNotIn("Cast", [node.op_type for node in onnx_program.model.graph])

    def test_onnx_export_conditional(self):
        # [FUNCTION DOC] Exports a model using a higher-order conditional, checks the graph
        # contains an If node, and verifies numerical agreement for an input taking the true
        # branch and one taking the false branch.
        class CondModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model choosing between adding one and subtracting
            # forty-two based on whether the input sum is positive.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Defines the two branch functions and returns the
                # result of the conditional on the sign of the sum.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch; returns the input plus one.
                    return x + 1.0

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch; returns the
                    # input minus forty-two.
                    return x - 42.0

                y = torch.cond(x.sum() > 0, true_fn, false_fn, [x])
                return y

        onnx_program = self.export(CondModel(), (torch.tensor([1, 2]),))
        onnx_model = onnx_program.model
        self.assertIn("If", [node.op_type for node in onnx_model.graph])
        onnx_testing.assert_onnx_program(onnx_program)
        # Test different branches
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([-1, -2]),))

    def test_onnx_export_nested_conditional_and_nested_weights(self):
        # [FUNCTION DOC] Exports a model with a conditional whose branches call a submodule
        # that also owns a conditional and weights, and verifies numerical agreement on
        # three inputs hitting different branches at both nesting levels.
        class Submodule(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule with one learnable weight whose forward runs a
            # conditional multiplying or dividing by that weight.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and registers a parameter
                # weight of value one hundred.
                super().__init__()
                # Nested weight
                self.weight = torch.nn.Parameter(torch.tensor([100.0]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs a conditional on whether the input sum is
                # non-positive, choosing between the nested branch functions, and
                # returns the result.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch; multiplies the input by
                    # the submodule weight.
                    return x * self.weight

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch; divides the input by
                    # the submodule weight.
                    return x / self.weight

                y = torch.cond(x.sum() <= 0, true_fn, false_fn, [x])
                return y

        class CondModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer model with its own weight and a Submodule; its
            # conditional branches use both.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module, creates the submodule and a
                # parameter weight of value forty-two.
                super().__init__()
                self.submodule = Submodule()
                self.weight = torch.nn.Parameter(torch.tensor([42.0]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs a conditional on whether the input sum is
                # positive; returns the result of the branch functions.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch; subtracts the outer weight from
                    # the input and passes it through the submodule.
                    return self.submodule(x - self.weight)

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch; subtracts the outer weight
                    # from the input.
                    return x - self.weight

                y = torch.cond(x.sum() > 0, true_fn, false_fn, [x])
                return y

        onnx_program = self.export(CondModel(), (torch.tensor([1, 2]),))
        onnx_testing.assert_onnx_program(onnx_program)
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([0, 0]),))
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([43, 43]),))

    def test_onnx_export_conditional_multi_outputs(self):
        # [FUNCTION DOC] Exports a conditional whose branches return two values and verifies
        # numerical agreement on inputs taking each branch.
        class CondModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose conditional returns a pair of updated tensors.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a ones tensor like the input and returns the
                # pair produced by the conditional, with the second returned item being the
                # original ones tensor.
                z = torch.ones_like(x)

                def true_fn(x, z):
                    # [FUNCTION DOC] HELPER: True branch; adds one to the first tensor,
                    # scales the second by one, and returns both.
                    x = x + 1.0
                    z = z * 1.0
                    return x, z

                def false_fn(x, z):
                    # [FUNCTION DOC] HELPER: False branch; subtracts one from the first
                    # tensor, zeroes the second, and returns both.
                    x = x - 1.0
                    z = z * 0.0
                    return x, z

                x = torch.cond(x.sum() > 0, true_fn, false_fn, (x, z))
                return x, z

        onnx_program = self.export(CondModel(), (torch.tensor([1, 2]),))
        onnx_testing.assert_onnx_program(onnx_program)
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.tensor([-1, -2]),))

    def test_onnx_export_while_loop_nested(self):
        # [FUNCTION DOC] Exports a model with a while loop whose body contains another while
        # loop, on four scalar tensor inputs, and verifies numerical agreement.
        class Nested(torch.nn.Module):
            # [CLASS DOC] HELPER: Model running a nested pair of higher-order while loops.
            def forward(self, ci, cj, a, b):
                # [FUNCTION DOC] HELPER: Defines loop condition and body functions (the body
                # runs an inner loop) and returns the final loop carried values.
                def cond_fn(i1, j1, x1, y1):
                    # [FUNCTION DOC] HELPER: Outer loop condition; true while the first
                    # counter is positive.
                    return i1 > 0

                def body_fn(i1, j1, x1, y1):
                    # [FUNCTION DOC] HELPER: Outer loop body; runs the inner loop, then
                    # decrements the first counter, clones the second, doubles the third and
                    # halves the fourth, returning all four.
                    def cond_fn_nested(i2, j2, x2, y2):
                        # [FUNCTION DOC] HELPER: Inner loop condition; true while the second
                        # counter is positive.
                        return j2 > 0

                    def body_fn_nested(i2, j2, x2, y2):
                        # [FUNCTION DOC] HELPER: Inner loop body; clones the first counter,
                        # decrements the second, adds 3.14 to the third and subtracts 2.71
                        # from the fourth.
                        return i2.clone(), j2 - 1, x2 + 3.14, y2 - 2.71

                    i1, j1, x1, y1 = torch.ops.higher_order.while_loop(
                        cond_fn_nested, body_fn_nested, [i1, j1, x1, y1], []
                    )
                    return i1 - 1, j1.clone(), x1 * 2, y1 / 2

                return torch.ops.higher_order.while_loop(
                    cond_fn, body_fn, [ci, cj, a, b], []
                )

        onnx_program = self.export(
            Nested(),
            (torch.tensor(2), torch.tensor(3), torch.tensor(1.0), torch.tensor(2.0)),
        )
        onnx_testing.assert_onnx_program(onnx_program)

    def test_onnx_export_while_loop_int_carry(self):
        # [FUNCTION DOC] Exports a while loop that carries a Python integer counter along
        # with a tensor, then verifies numerical agreement.
        class IntCarry(torch.nn.Module):
            # [CLASS DOC] HELPER: Model running a while loop with an integer counter
            # and a tensor, then returning derived values including a tensor sized from
            # the final counter.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the while loop operator directly, adds range
                # checks so the final counter can size a ones tensor, and returns four
                # values derived from the counter and the carried tensor.
                def cond_fn(it, x):
                    # [FUNCTION DOC] HELPER: Loop condition; true while the counter is below
                    # the first dimension size of the tensor.
                    return it < x.shape[0]

                def body_fn(it, x):
                    # [FUNCTION DOC] HELPER: Loop body; asserts the counter is a valid
                    # index, adds the counter to the selected row of a clone of the tensor,
                    # and returns the incremented counter with the clone.
                    x_clone = x.clone()
                    # Need these checks to select from x
                    torch._check(it >= 0)
                    torch._check(it < x.shape[0])
                    x_clone.select(0, it).copy_(x_clone.select(0, it) + it)
                    return it + 1, x_clone

                # We invoke the hop directly to avoid triggering dynamo tracing
                out_it, out_x = torch.ops.higher_order.while_loop(
                    cond_fn, body_fn, (0, x), tuple()
                )
                # We need torch._check to use it in torch.ones call
                torch._check(out_it > 0)
                return (
                    out_it + 1,
                    out_it + out_x,
                    out_it < x.shape[0],
                    torch.ones(out_it * 2),
                )

        onnx_program = self.export(
            IntCarry(),
            (torch.tensor([10.0, 20.0, 30.0]),),
        )
        onnx_testing.assert_onnx_program(onnx_program)

    def test_empty(self):
        # [FUNCTION DOC] Only checks that export of a model creating an uninitialized
        # integer tensor from the input size succeeds; no numerical comparison is possible.
        class EmptyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning an uninitialized 64-bit integer tensor
            # shaped like the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns an empty integer tensor with
                # the input size.
                return torch.empty(x.size(), dtype=torch.int64)

        # Since `torch.empty` returns tensor with uninitialized data, we cannot
        # test this under `test_fx_to_onnx_with_onnxruntime.py` with result comparison.
        _ = self.export(EmptyModel(), (torch.randn(1, 2),))

    def test_multiple_outputs_op_with_evaluator(self):
        # [FUNCTION DOC] Exports a model summing the top three values of a
        # gradient-requiring input (an operator with multiple outputs) and
        # checks numerical agreement.
        class TopKModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model summing the three largest input values.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Takes the top three values and returns their sum.
                values, _ = torch.topk(x, 3)
                return torch.sum(values)

        onnx_program = self.export(
            TopKModel(), (torch.arange(1.0, 6.0, requires_grad=True),)
        )
        onnx_testing.assert_onnx_program(onnx_program)

    def test_exported_program_torch_distributions_normal_Normal(self):
        # [FUNCTION DOC] Exports a model sampling from a normal distribution, first via the
        # torch exporter under no-grad in non-strict mode, then passes the resulting
        # exported program to the ONNX export. Only checks that no error is raised.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model holding a standard normal distribution and sampling
            # a tensor of the input shape.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the standard normal distribution before
                # initializing the base module.
                self.normal = torch.distributions.normal.Normal(0, 1)
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a sample from the distribution with the
                # shape of the input.
                return self.normal.sample(x.shape)

        with torch.no_grad():
            exported_program = torch.export.export(
                Model(), args=(torch.randn(2),), strict=False
            )
            _ = self.export(exported_program)

    @common_utils.parametrize(
        "float8_type, onnx_type",
        [
            common_utils.subtest(
                (torch.float8_e5m2, ir.DataType.FLOAT8E5M2),
                name="torch_float8_e5m2",
            ),
            common_utils.subtest(
                (torch.float8_e5m2fnuz, ir.DataType.FLOAT8E5M2FNUZ),
                name="torch_float8_e5m2fnuz",
            ),
            common_utils.subtest(
                (torch.float8_e4m3fn, ir.DataType.FLOAT8E4M3FN),
                name="torch_float8_e4m3fn",
            ),
            common_utils.subtest(
                (torch.float8_e4m3fnuz, ir.DataType.FLOAT8E4M3FNUZ),
                name="torch_float8_e4m3fnuz",
            ),
        ],
    )
    def test_float8_support(self, float8_type: torch.dtype, onnx_type: ir.DataType):
        # [FUNCTION DOC] Parametrized over four float8 variants (e5m2, e5m2fnuz, e4m3fn,
        # e4m3fnuz). Exports a model casting a float input to the given float8 type and
        # checks the first graph output has the matching ONNX data type.
        class Float8Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Model casting its input to the float8 type chosen by
            # the enclosing test.
            def forward(self, input: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns the input converted to the
                # captured float8 type.
                input = input.to(float8_type)
                return input

        onnx_program = self.export(Float8Module(), (torch.randn(1, 2),))
        self.assertEqual(onnx_program.model.graph.outputs[0].dtype, onnx_type)

    def test_float4_support(self):
        # [FUNCTION DOC] Exports a model with no inputs returning an empty packed float4
        # tensor, without optimization, and checks the output has the ONNX float4 type and
        # an unpacked shape of two.
        class Float4Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Model without inputs returning an empty packed float4
            # tensor of one element.
            def forward(self):
                # [FUNCTION DOC] HELPER: Returns an empty packed float4 tensor with a
                # single (two-value) element.
                return torch.empty([1], dtype=torch.float4_e2m1fn_x2)

        onnx_program = self.export(Float4Module(), optimize=False)
        output = onnx_program.model.graph.outputs[0]
        self.assertEqual(output.dtype, ir.DataType.FLOAT4E2M1)
        # The shape is [*shape[:-1], shape[-1]*2] because ONNX stores the shape of the unpacked tensor
        self.assertEqual(output.shape.numpy(), [2])

    def test_bfloat16_support(self):
        # [FUNCTION DOC] Exports a model with a bfloat16 parameter and a bfloat16 constant
        # without optimization, and checks there are two initializers, all bfloat16, and
        # that the graph input and output are bfloat16.
        class BfloatModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a bfloat16 scalar parameter whose forward
            # multiplies the input by a bfloat16 constant and the parameter.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and registers a bfloat16
                # parameter of value two.
                super().__init__()
                # Test parameters
                self.param = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.bfloat16))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiplies the input by a bfloat16 constant of one
                # and by the parameter, and returns it.
                # Test constant tensors are stored as bfloat16
                const = torch.tensor(1.0, dtype=torch.bfloat16)
                return x * const * self.param

        input = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        onnx_program = self.export(BfloatModel(), (input,), optimize=False)
        initializers = onnx_program.model.graph.initializers.values()
        self.assertEqual(len(initializers), 2)
        for initializer in initializers:
            self.assertEqual(initializer.dtype, ir.DataType.BFLOAT16)
        self.assertEqual(onnx_program.model.graph.inputs[0].dtype, ir.DataType.BFLOAT16)
        self.assertEqual(
            onnx_program.model.graph.outputs[0].dtype, ir.DataType.BFLOAT16
        )

    def test_export_with_logging_logger(self):
        # [FUNCTION DOC] Exports a model that logs through the standard library logger in
        # forward and checks numerical agreement.
        logger = logging.getLogger(__name__)

        class LoggingLoggerModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Model logging an info message through a standard logger
            # and adding one to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Logs the message at info level using the logger of
                # the enclosing test, then returns the input plus one.
                logger.info("abc")
                return x + 1

        onnx_program = self.export(LoggingLoggerModule(), (torch.tensor(1),))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_export_with_hf_logging_logger(self):
        # [FUNCTION DOC] Exports a model that logs through the transformers logging helper
        # (warn once) in forward and checks numerical agreement.
        logger = transformers.utils.logging.get_logger(__name__)

        class HFLoggingLoggerModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Model logging a warning once through the transformers
            # logger and adding one to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Emits the warn-once message using the enclosing
                # test logger, then returns the input plus one.
                logger.warning_once("abc")
                return x + 1

        onnx_program = self.export(HFLoggingLoggerModule(), (torch.tensor(1),))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_export_with_print(self):
        # [FUNCTION DOC] Exports a model that prints in forward and checks numerical
        # agreement, i.e. that a print call in forward does not break export.
        class PrintModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Model printing a message and adding one to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Prints a message, then returns the input plus one.
                print("abc")
                return x + 1

        onnx_program = self.export(PrintModule(), (torch.tensor(1),))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_export_with_dynamic_input(self):
        # [FUNCTION DOC] Exports with a dynamic first dimension on the input and checks
        # numerical agreement on an input with a different first dimension size.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding one to its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus 1.0.
                return x + 1.0

        dim0 = torch.export.Dim("dim0")
        onnx_program = self.export(
            Model(),
            (torch.randn(2, 3, 4, dtype=torch.float),),
            dynamic_shapes=({0: dim0},),
        )

        onnx_testing.assert_onnx_program(
            onnx_program, args=(torch.randn(3, 3, 4, dtype=torch.float),)
        )

    def test_dynamic_shape_bilstm_batch_first(self):
        # [FUNCTION DOC] Exports a bidirectional batch-first LSTM without optimization,
        # with a dynamic time dimension named time on the input named x. Checks the first
        # graph input's second dimension is the symbol time, then checks numerical
        # agreement on a longer sequence.
        class BiLSTM(torch.nn.Module):
            # [CLASS DOC] HELPER: Model wrapping one bidirectional batch-first LSTM (input
            # size four, hidden size three) and returning only the output sequence.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and creates the LSTM.
                super().__init__()
                self.lstm = torch.nn.LSTM(4, 3, batch_first=True, bidirectional=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the LSTM and returns its output, dropping
                # the final states.
                out, _ = self.lstm(x)
                return out

        time = torch.export.Dim("time")
        onnx_program = self.export(
            BiLSTM(),
            (torch.randn(2, 5, 4),),
            input_names=["x"],
            dynamic_shapes={"x": {1: time}},
            optimize=False,
        )
        self.assertEqual(onnx_program.model.graph.inputs[0].shape[1].value, "time")
        onnx_testing.assert_onnx_program(onnx_program, args=(torch.randn(2, 7, 4),))

    def test_export_with_specialized_input_during_tracing(self):
        # [FUNCTION DOC] Exports with a dynamic first dimension (minimum six) for the first
        # input and a static second input of value five, traced with a plain integer for the
        # second input, then checks numerical agreement on a different first dimension with
        # the same integer.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
                return x + y

        dim0_x = torch.export.Dim("dim0_x", min=6)
        dynamic_shapes = {"x": {0: dim0_x}, "y": torch.export.Dim.STATIC}
        # specialized input y to 5 during tracing
        onnx_program = self.export(
            Model(),
            (
                torch.ones(7, 5),
                5,
            ),
            dynamic_shapes=dynamic_shapes,
        )

        onnx_testing.assert_onnx_program(onnx_program, args=(torch.ones(8, 5), 5))

    def test_export_with_none_arg_name_in_dynamic(self):
        # [FUNCTION DOC] Exports with a tuple of dynamic shape specs where the first input
        # has none (static) and the second has a dynamic first dimension, then checks
        # numerical agreement on a second input with a different first dimension.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the sum of the sums of both inputs.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the sum of all elements of the first input
                # plus that of the second.
                return a.sum() + b.sum()

        dim = torch.export.Dim("dim")
        onnx_program = self.export(
            Model(),
            (
                torch.randn(4, 4),
                torch.randn(4, 4),
            ),
            dynamic_shapes=(None, {0: dim}),
        )

        test_inputs = (
            torch.randn(4, 4),
            torch.randn(7, 4),
        )
        onnx_testing.assert_onnx_program(onnx_program, args=test_inputs)

    def test_export_with_non_arg_name_with_kwarg(self):
        # [FUNCTION DOC] Exports with positional arguments plus keyword arguments passed in
        # a different order than the signature, with a dynamic shape tuple following the
        # signature order, then checks numerical agreement with changed dimensions and the
        # keyword order flipped.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model combining the sums of two positional and
            # two keyword inputs.
            def forward(self, a, b, kw1, kw2):
                # [FUNCTION DOC] HELPER: Returns the sum of the first two inputs plus the
                # sum of kw1 minus the sum of kw2.
                return a.sum() + b.sum() + kw1.sum() - kw2.sum()

        dim = torch.export.Dim("dim")
        dim_for_kw1 = torch.export.Dim("dim_for_kw1")
        onnx_program = self.export(
            Model(),
            (torch.randn(4, 4), torch.randn(4, 4)),
            {"kw2": torch.ones(4, 4), "kw1": torch.zeros(4, 4)},
            # Tuple dynamic_shapes follows the original function signature order.
            dynamic_shapes=(None, {0: dim}, {0: dim_for_kw1}, None),
        )

        # This should work even if the kwarg order are flipped.
        onnx_testing.assert_onnx_program(
            onnx_program,
            args=(torch.randn(4, 4), torch.randn(7, 4)),
            kwargs={"kw2": torch.ones(4, 4), "kw1": torch.zeros(9, 4)},
        )

    def test_export_with_input_lifting_buffers_mutation(self):
        # [FUNCTION DOC] Runs twice, with a persistent and with a non-persistent buffer.
        # Each time defines a model that mutates a buffer in place in forward, exports it
        # with a dynamic first dimension on both inputs, and checks numerical agreement
        # on larger inputs.
        for persistent in (True, False):

            class CustomModule(torch.nn.Module):
                # [CLASS DOC] HELPER: Model with a scalar buffer (persistence chosen by the
                # loop variable) that is incremented in place during forward.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Initializes the module and registers the
                    # buffer my_buffer with value four, persistent or not as chosen by
                    # the enclosing loop.
                    super().__init__()
                    self.register_buffer(
                        "my_buffer", torch.tensor(4.0), persistent=persistent
                    )

                def forward(self, x, b):
                    # [FUNCTION DOC] HELPER: Adds the two inputs for the result, mutates
                    # the buffer by adding one in place (discarding that value), and
                    # returns the result.
                    output = x + b
                    (
                        self.my_buffer.add_(1.0) + 3.0
                    )  # Mutate buffer through in-place addition
                    return output

            dim = torch.export.Dim("dim")
            onnx_program = self.export(
                CustomModule(),
                (
                    torch.rand((3, 3), dtype=torch.float32),
                    torch.randn(3, 3),
                ),
                dynamic_shapes=({0: dim}, {0: dim}),
            )

            onnx_testing.assert_onnx_program(
                onnx_program,
                args=(torch.rand((4, 3), dtype=torch.float32), torch.randn(4, 3)),
            )

    def test_export_with_non_arg_name_with_container_type(self):
        # [FUNCTION DOC] Exports a model taking a pair of tensors and a tensor, with dynamic
        # shapes built by mapping over the input structure so only the second leaf of the
        # pair gets a dynamic first dimension (minimum three). Checks numerical agreement
        # with a differently sized second leaf, keeping the original input structure.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model summing the elements of both members of a pair and
            # a third tensor.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the sum of the first pair member, plus the
                # second pair member, plus the second argument.
                return a[0].sum() + a[1].sum() + b.sum()

        count = 0

        def dynamify_inp(x):
            # [FUNCTION DOC] HELPER: Used as a mapping function over input leaves. Counts
            # calls in an enclosing counter; the leaf visited second gets a dynamic first
            # dimension (minimum three), all others get none. Mutates the counter.
            # Mark the second input a[1] dynamic
            nonlocal count
            if count == 1:
                dim = torch.export.Dim("dim", min=3)
                count += 1
                return {0: dim}
            count += 1
            return None

        dynamic_shapes = torch_pytree.tree_map(
            dynamify_inp,
            (
                (torch.randn(4, 4), torch.randn(4, 4)),
                torch.randn(4, 4),
            ),
        )
        onnx_program = self.export(
            Model(),
            (
                (torch.randn(4, 4), torch.randn(4, 4)),
                torch.randn(4, 4),
            ),
            dynamic_shapes=dynamic_shapes,
        )

        # NOTE: Careful with the input format. The input format should be
        # consistent with how the model is exported.
        onnx_testing.assert_onnx_program(
            onnx_program,
            args=((torch.randn(4, 4), torch.randn(6, 4)), torch.randn(4, 4)),
        )

    def test_export_with_lazy_module_kwargs(self):
        # [FUNCTION DOC] Exports a lazy module (parameters initialized on first call) with
        # only keyword arguments and a dynamic first dimension for both, then checks
        # numerical agreement with keyword inputs of a different size.
        class LazyModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            # [CLASS DOC] HELPER: Lazy-module model adding two inputs; the lazy
            # initialization hook does nothing.
            def initialize_parameters(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Lazy-initialization hook that
                # intentionally does nothing.
                pass

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
                return x + y

        m = LazyModule()
        dim = torch.export.Dim("dim")
        dynamic_shapes = ({0: dim}, {0: dim})
        onnx_program = self.export(
            m,
            (),
            {"x": torch.randn(3, 3), "y": torch.randn(3, 3)},
            dynamic_shapes=dynamic_shapes,
        )

        inputs = {"x": torch.randn(6, 3), "y": torch.randn(6, 3)}
        onnx_testing.assert_onnx_program(onnx_program, kwargs=inputs)

    def test_export_of_rename_dynamic_axes_required_model_with_mixed_type_of_dynamic_shapes(
        self,
    ):
        # [FUNCTION DOC] Exports a model with nested list and dictionary inputs using mixed
        # dynamic shape specs: a named dimension, custom string axis names, and automatic
        # dimensions. Expects a user warning that some axis names are unused because they
        # share shape constraints with another axis. Then checks numerical agreement on
        # smaller inputs and that the first dimension names of the graph inputs are the
        # dimension x name for the first five inputs and the custom c name for the last.
        class NestedModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model taking a tensor, a list of two tensors, a
            # dictionary of two tensors and a tensor; shifts or subtracts by constants
            # based on shape conditions.
            def forward(
                self,
                x: torch.Tensor,
                ys: list[torch.Tensor],
                zs: dict[str, torch.Tensor],
                c: torch.Tensor,
            ):
                # [FUNCTION DOC] HELPER: Sums the list items and dictionary entries into
                # one tensor. If the first input has fewer than three elements and the
                # last input does not have four, returns the first plus five, the first
                # plus the sum, and the last; otherwise returns the matching subtractions
                # and the last.
                y = ys[0] + ys[1] + zs["a"] + zs["b"]
                w = 5
                if x.shape[0] < 3 and c.shape[0] != 4:
                    return x + w, x + y, c
                else:
                    return x - w, x - y, c

        input = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(6),
        )

        dynamic_shapes = (
            {0: torch.export.Dim("dim_x", min=3)},  # Dim
            [("custom_name_axis_ys_0",), (torch.export.Dim.AUTO,)],  # custom name
            {
                "a": {0: torch.export.Dim.AUTO},
                "b": ("custom_name_axis_zs_b_0",),
            },  # _DimHint
            {0: "custom_name_axis_c_0"},  # custom name
        )

        # 0. Export the model
        # 1. Assert the warning message
        with self.assertWarnsRegex(
            UserWarning,
            "# The axis name: .* will not be used, since it shares the same shape constraints with another axis: .*.",
        ):
            onnx_program = self.export(
                NestedModel(), input, dynamic_shapes=dynamic_shapes, optimize=False
            )
        # 2. Assert the exported model
        input = (
            torch.ones(4),
            [torch.zeros(4), torch.ones(4)],
            {"a": torch.zeros(4), "b": torch.ones(4)},
            torch.ones(5),
        )
        onnx_testing.assert_onnx_program(
            onnx_program,
            args=input,
        )
        # 3. Assert the dynamic axes names
        # Some names are not respected because they share the same shape constraints,
        # so they are the same to ExportedProgram.
        expected_axis_names = [
            "dim_x",
            "dim_x",
            "dim_x",
            "dim_x",
            "dim_x",
            "custom_name_axis_c_0",
        ]

        for expected_axis_name, input in zip(
            expected_axis_names, onnx_program.model.graph.inputs
        ):
            self.assertEqual(input.shape[0].value, expected_axis_name)

    def test_export_of_static_dim_constraints(self):
        # [FUNCTION DOC] Exports a model with a linear layer using dynamic shape specs
        # mixing bounded named dimensions (one derived as another plus one), automatic and
        # static markers, and a fixed integer size. Checks numerical agreement and that the
        # first graph input's first dimension is named dx.
        # NOTE: This test is to ensure that the static dim constraints are respected.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a linear layer (six to four features) applied
            # to the first input, added to the second input without its first row; also
            # returns the third input doubled.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and creates
                # the linear layer.
                super().__init__()
                self.l = torch.nn.Linear(6, 4)

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns a pair: the linear output of the first
                # input plus the second input from index one onward, and the third
                # input times two.
                x0 = self.l(x) + y[1:]
                return x0, z * 2.0

        inputs = (torch.randn(4, 6), torch.randn(5, 4), torch.randn(3, 3))
        dx = torch.export.Dim("dx", min=3, max=6)
        dy = dx + 1
        dz = torch.export.Dim("dz", min=3, max=6)

        # all of these should be fine
        dynamic_shapes = (
            {0: dx, 1: torch.export.Dim.AUTO},
            {0: dy, 1: torch.export.Dim.STATIC},
            {0: dz, 1: 3},
        )
        onnx_program = self.export(Model(), inputs, dynamic_shapes=dynamic_shapes)
        onnx_testing.assert_onnx_program(onnx_program)
        # make sre the naming is working
        self.assertEqual(onnx_program.model.graph.inputs[0].shape[0], "dx")

    def test_export_sym_max(self):
        # [FUNCTION DOC] Exports a model returning the symbolic maximum of the two dimension
        # sizes, with both dimensions dynamic, checks numerical agreement and that the graph
        # has a Max node.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the maximum of the input's
            # two dimension sizes.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the symbolic maximum of the
                # input shape entries.
                return torch.sym_max(*x.shape)

        inputs = (torch.zeros((2, 3)),)
        dynamic_shapes = ({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},)
        onnx_program = self.export(Model(), inputs, dynamic_shapes=dynamic_shapes)
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertIn(
            "Max",
            [node.op_type for node in onnx_program.model.graph],
        )

    def test_export_sym_min(self):
        # [FUNCTION DOC] Exports a model returning the symbolic minimum of the two dimension
        # sizes, with both dimensions dynamic, checks numerical agreement and that the graph
        # has a Min node.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the minimum of the input's
            # two dimension sizes.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the symbolic minimum of the
                # input shape entries.
                return torch.sym_min(*x.shape)

        inputs = (torch.zeros((2, 3)),)
        dynamic_shapes = ({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},)
        onnx_program = self.export(Model(), inputs, dynamic_shapes=dynamic_shapes)
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertIn(
            "Min",
            [node.op_type for node in onnx_program.model.graph],
        )

    def test_export_sym_not(self):
        # [FUNCTION DOC] Exports a model returning the symbolic negation of an equality
        # test of two dynamic dimensions, checks numerical agreement and that the graph has
        # a Not node.
        class SymNotModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning whether the two dimensions differ, via the
            # symbolic not operation.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Compares the two dimension sizes for equality and
                # returns the symbolic negation.
                comparison = x.shape[0] == x.shape[1]
                return torch.sym_not(comparison)

        inputs = (torch.zeros((2, 2)),)
        dynamic_shapes = ({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},)
        onnx_program = self.export(SymNotModel(), inputs, dynamic_shapes=dynamic_shapes)
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertIn(
            "Not",
            [node.op_type for node in onnx_program.model.graph],
        )

    def test_export_sym_float(self):
        # [FUNCTION DOC] Exports a model converting a dynamic dimension size to a symbolic
        # float, checks numerical agreement and that the graph has a Cast node.
        class SymFloatModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the first dimension size as
            # a symbolic float.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Converts the first dimension size to a symbolic
                # float and returns it.
                a = x.shape[0]
                return torch.sym_float(a)

        inputs = (torch.zeros((2, 2)),)
        dynamic_shapes = ({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},)
        onnx_program = self.export(
            SymFloatModel(), inputs, dynamic_shapes=dynamic_shapes
        )
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertIn(
            "Cast",
            [node.op_type for node in onnx_program.model.graph],
        )

    def test_export_sym_sum(self):
        # [FUNCTION DOC] Exports a model returning the symbolic sum of the dimension sizes
        # of a three dimensional input (first two dimensions dynamic), checks numerical
        # agreement and that the graph has an Add node.
        class SymSumModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the symbolic sum of all dimension sizes
            # of the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the symbolic sum of the input shape.
                return torch.sym_sum(x.shape)

        inputs = (torch.zeros((2, 3, 4)),)
        dynamic_shapes = ({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},)
        onnx_program = self.export(SymSumModel(), inputs, dynamic_shapes=dynamic_shapes)
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertIn(
            "Add",
            [node.op_type for node in onnx_program.model.graph],
        )

    def test_export_sym_ite(self):
        # [FUNCTION DOC] Exports a model using the symbolic if-then-else on whether the
        # first dimension exceeds the second, checks numerical agreement and that the graph
        # has a Where node.
        class SymIteModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the larger of the two dimension sizes
            # through a symbolic conditional.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Tests whether the first dimension is larger
                # than the second and returns the first if so, otherwise the second,
                # using the symbolic conditional.
                condition = x.shape[0] > x.shape[1]
                return torch.sym_ite(condition, x.shape[0], x.shape[1])

        inputs = (torch.zeros((3, 2)),)
        dynamic_shapes = ({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},)
        onnx_program = self.export(SymIteModel(), inputs, dynamic_shapes=dynamic_shapes)
        onnx_testing.assert_onnx_program(onnx_program)
        self.assertIn(
            "Where",
            [node.op_type for node in onnx_program.model.graph],
        )

    def test_scan_cdist_add(self):
        # [FUNCTION DOC] Exports a model that calls the higher-order scan over the rows of a
        # tensor with a zero initial carry and the full tensor as an additional input, and
        # checks numerical agreement.
        def dist(unused: torch.Tensor, x: torch.Tensor, samex: torch.Tensor):
            # [FUNCTION DOC] HELPER: Scan body. Computes, for the current row, the Euclidean
            # distance to each row of the additional input and returns a list of the cloned
            # carry and the distance vector.
            sub = samex - x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            return [unused.clone(), rd]

        class ScanModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model running the scan with the module-level distance
            # function and returning the stacked distances.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Makes a zero carry and a clone of the input, runs
                # the scan over the input rows, and returns the second scan output (the
                # stacked per-row distances).
                z = torch.tensor([0], dtype=torch.float32)
                y = x.clone()
                out = torch.ops.higher_order.scan(dist, [z], [x], additional_inputs=[y])
                return out[1]

        inputs = (
            torch.tensor(
                [[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32
            ),
        )
        onnx_program = self.export(ScanModel(), inputs)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_scan_cdist_dynamic_shapes(self):
        # [FUNCTION DOC] Same scan-based distance computation as the previous test but with
        # two inputs with independent dynamic row counts and a shared dynamic column count;
        # checks numerical agreement.
        def dist(y: torch.Tensor, scanned_x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Scan body. Computes the Euclidean distance between the
            # carried tensor and the current row, and returns a list of the cloned carry and
            # the distance vector.
            sub = y - scanned_x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            return [y.clone(), rd]

        class ScanModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model scanning the first input's rows with the second
            # input as the carry, returning only the stacked outputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Runs the scan with the second input as carry
                # and the first as scanned input, ignores the final carry and returns
                # the stacked outputs.
                carry, out = torch.ops.higher_order.scan(
                    dist, [y], [x], additional_inputs=[]
                )
                return out

        x_rows = torch.export.Dim("x_rows")
        y_rows = torch.export.Dim("y_rows")
        dim = torch.export.Dim("dim")
        inputs = (torch.randn(3, 4), torch.randn(5, 4))
        onnx_program = self.export(
            ScanModel(),
            inputs,
            dynamic_shapes=({0: x_rows, 1: dim}, {0: y_rows, 1: dim}),
        )
        onnx_testing.assert_onnx_program(onnx_program)

    @pytest.mark.xfail(reason="Data dependent error.")
    def test_scan_loop_inplace(self):
        # [FUNCTION DOC] Expected to fail (marked xfail with reason data dependent error).
        # Builds an exported program for a model that, when exporting, replaces a Python
        # loop copying leading entries of each row by a scan, with dynamic dimensions, then
        # exports it to ONNX and checks numerical agreement.
        def dummy_loop(padded: torch.Tensor, pos: torch.Tensor):
            # [FUNCTION DOC] HELPER: Eager reference. For each row, copies the first pos
            # entries of the padded row into a zero tensor of the same shape and returns it.
            copy = torch.zeros(padded.shape)
            for i in range(pos.shape[0]):
                p = pos[i]
                copy[i, :p] = padded[i, :p]
            return copy

        def dummy_loop_with_scan(padded: torch.Tensor, pos: torch.Tensor):
            # [FUNCTION DOC] HELPER: Scan-based version of the loop. Scans over the rows and
            # their lengths with no carry, and returns the scan result.
            def pad_row(padded, p):
                # [FUNCTION DOC] HELPER: Scan body. Creates a zero row, adds range checks on
                # the length (plus an extra check only while exporting), copies the leading
                # entries from the input row, and returns the row in a tuple.
                row = torch.zeros((padded.shape[0],))
                torch._check(p.item() > 0)
                torch._check(p.item() < padded.shape[0])
                # this check is not always true, we add it anyway to make this dimension >= 2
                # and avoid raising an exception about dynamic dimension in {0, 1}
                if torch.compiler.is_exporting():
                    torch._check(p.item() > 1)
                row[: p.item()] = padded[: p.item()]
                return (row,)

            return torch.ops.higher_order.scan(pad_row, [], [padded, pos], [])

        def select_when_exporting(f, f_scan):
            # [FUNCTION DOC] HELPER: Returns the scan version when the program is being
            # exported, otherwise the plain loop version.
            return f_scan if torch.compiler.is_exporting() else f

        class ScanModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying the loop variant chosen by export state to
            # its images and positions.
            def forward(self, images, position):
                # [FUNCTION DOC] HELPER: Selects the loop implementation (scan while
                # exporting) and applies it to the images and position tensors.
                return select_when_exporting(dummy_loop, dummy_loop_with_scan)(
                    images, position
                )

        DYN = torch.export.Dim.DYNAMIC
        x = torch.randn((5, 6))
        y = torch.arange(5, dtype=torch.int64) + 1
        ep = torch.export.export(
            ScanModel(),
            (x, y),
            dynamic_shapes={"images": {0: DYN, 1: DYN}, "position": {0: DYN}},
            strict=False,
        )
        onnx_program = self.export(ep)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_complex_initializer(self):
        # [FUNCTION DOC] Exports a model with a complex buffer, a complex parameter and a
        # complex constant created in forward, and checks numerical agreement.
        class ComplexInitModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model holding a complex buffer and a complex parameter and
            # combining them with a complex constant and the input.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and registers a
                # three-element complex buffer and a three-element complex parameter.
                super().__init__()
                self.buffer = torch.nn.parameter.Buffer(
                    torch.complex(
                        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])
                    )
                )
                self.weight = torch.nn.Parameter(
                    torch.complex(
                        torch.tensor([7.0, 8.0, 9.0]), torch.tensor([10.0, 11.0, 12.0])
                    )
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a complex constant, adds it and the buffer
                # to the input, and multiplies by the parameter.
                constant = torch.complex(
                    torch.tensor([11.0, 12.0, 13.0]), torch.tensor([14.0, 15.0, 16.0])
                )
                return (x + constant + self.buffer) * self.weight

        x = torch.complex(
            torch.tensor([10.0, 20.0, 30.0]), torch.tensor([40.0, 50.0, 60.0])
        )

        onnx_program = self.export(ComplexInitModel(), (x,))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_adaptive_max_pool2d_global_pooling(self):
        # [FUNCTION DOC] Exports a two dimensional adaptive max pooling to output size one
        # by one, on a 1x3x8x8 input, and checks numerical agreement.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model wrapping an adaptive max pooling layer to
            # 1x1 spatial output.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and creates
                # the pooling layer.
                super().__init__()
                self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the pooling layer applied to the input.
                return self.pool(x)

        x = torch.randn(1, 3, 8, 8)
        onnx_program = self.export(Model(), (x,))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_adaptive_max_pool2d_evenly_divisible(self):
        # [FUNCTION DOC] Exports a two dimensional adaptive max pooling to a 2x2 output on a
        # 4x4 input (evenly divisible), and checks numerical agreement.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model wrapping an adaptive max pooling layer to a
            # 2x2 spatial output.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and creates
                # the pooling layer.
                super().__init__()
                self.pool = torch.nn.AdaptiveMaxPool2d((2, 2))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the pooling layer applied to the input.
                return self.pool(x)

        x = torch.randn(1, 3, 4, 4)
        onnx_program = self.export(Model(), (x,))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_adaptive_max_pool3d_global_pooling(self):
        # [FUNCTION DOC] Exports a three dimensional adaptive max pooling to a 1x1x1 output
        # on a 4x4x4 volume, and checks numerical agreement.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model wrapping a three dimensional adaptive max pooling
            # layer to a single-cell output.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and creates
                # the pooling layer.
                super().__init__()
                self.pool = torch.nn.AdaptiveMaxPool3d((1, 1, 1))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the pooling layer applied to the input.
                return self.pool(x)

        x = torch.randn(1, 3, 4, 4, 4)
        onnx_program = self.export(Model(), (x,))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_adaptive_max_pool3d_evenly_divisible(self):
        # [FUNCTION DOC] Exports a three dimensional adaptive max pooling to a 2x2x2 output
        # on a 4x4x4 volume (evenly divisible), and checks numerical agreement.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model wrapping a three dimensional adaptive max pooling
            # layer to a 2x2x2 output.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and creates
                # the pooling layer.
                super().__init__()
                self.pool = torch.nn.AdaptiveMaxPool3d((2, 2, 2))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the pooling layer applied to the input.
                return self.pool(x)

        x = torch.randn(1, 3, 4, 4, 4)
        onnx_program = self.export(Model(), (x,))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_adaptive_max_pool1d_global_pooling(self):
        # [FUNCTION DOC] Exports a one dimensional adaptive max pooling to output length one
        # on an input of length eight, and checks numerical agreement.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model wrapping a one dimensional adaptive max pooling
            # layer to length one.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and creates
                # the pooling layer.
                super().__init__()
                self.pool = torch.nn.AdaptiveMaxPool1d(1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the pooling layer applied to the input.
                return self.pool(x)

        x = torch.randn(1, 3, 8)
        onnx_program = self.export(Model(), (x,))
        onnx_testing.assert_onnx_program(onnx_program)

    def test_adaptive_max_pool1d_evenly_divisible(self):
        # [FUNCTION DOC] Exports a one dimensional adaptive max pooling to output length two
        # on an input of length four (evenly divisible), and checks numerical agreement.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model wrapping a one dimensional adaptive max pooling
            # layer to length two.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and creates
                # the pooling layer.
                super().__init__()
                self.pool = torch.nn.AdaptiveMaxPool1d(2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the pooling layer applied to the input.
                return self.pool(x)

        x = torch.randn(1, 3, 4)
        onnx_program = self.export(Model(), (x,))
        onnx_testing.assert_onnx_program(onnx_program)


@common_utils.instantiate_parametrized_tests
class DynamoExporterNewOpsetsTest(common_utils.TestCase, _WithExport):
    # [CLASS DOC] Test case for exports that target newer ONNX opset versions (21 and 23);
    # mixes in the export helper and is expanded by the parametrization decorator.
    def test_group_norm_opset_21(self):
        # [FUNCTION DOC] Exports group normalization at opset 21 and checks the graph
        # contains a GroupNormalization node. No numerical comparison is made because the
        # runtime lacked the operator.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying group normalization with four groups.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the group normalization of the input
                # with four groups.
                return torch.nn.functional.group_norm(x, 4)

        x = torch.randn(1, 4, 4, 4, dtype=torch.float32)
        onnx_program = self.export(Model(), (x,), opset_version=21)
        # TODO(after ort support): As of ONNX Runtime 1.22, the operator is not implemented yet.
        # call assert_onnx_program after ort support
        self.assertIn(
            "GroupNormalization",
            [node.op_type for node in onnx_program.model.graph],
        )

    def test_attention_opset_23(self):
        # [FUNCTION DOC] Exports scaled dot product attention at opset 23 for half precision
        # inputs, checks the graph consists of exactly one Attention node, and checks
        # numerical agreement with loose tolerances.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model computing scaled dot product attention.
            def forward(self, query, key, value):
                # [FUNCTION DOC] HELPER: Returns scaled dot product attention of query,
                # key and value.
                return torch.nn.functional.scaled_dot_product_attention(
                    query, key, value
                )

        query = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16)
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16)

        onnx_program = self.export(Model(), (query, key, value), opset_version=23)
        self.assertEqual(["Attention"], [n.op_type for n in onnx_program.model.graph])

        onnx_testing.assert_onnx_program(onnx_program, atol=1e-2, rtol=1)

    def test_rms_norm(self):
        # [FUNCTION DOC] Exports RMS normalization over the last dimension and then over the
        # last two dimensions at opset 23, checking numerical agreement for each.
        """Test RMS normalization with various configurations."""

        class RMSNormModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying RMS normalization over a trailing dimension
            # of size three.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns RMS normalization of the input over
                # its last dimension.
                return torch.nn.functional.rms_norm(x, [3])

        x = torch.randn(2, 5, 3)
        onnx_program = self.export(RMSNormModel(), (x,), opset_version=23)
        onnx_testing.assert_onnx_program(onnx_program)

        # Test with multi-dimensional normalized_shape
        class RMSNormModel2D(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying RMS normalization over trailing dimensions
            # of sizes seven and three.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns RMS normalization of the input over its
                # last two dimensions.
                return torch.nn.functional.rms_norm(x, [7, 3])

        x = torch.randn(2, 5, 7, 3)
        onnx_program = self.export(RMSNormModel2D(), (x,), opset_version=23)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_rms_norm_with_weight(self):
        # [FUNCTION DOC] Exports RMS normalization with a learnable weight at opset 23 and
        # checks numerical agreement.
        """Test RMS normalization with weight parameter."""

        class RMSNormWithWeight(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a weight of ones, applying RMS normalization
            # with that weight.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and creates a three-element
                # parameter of ones.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns RMS normalization of the input over the
                # last dimension scaled by the weight.
                return torch.nn.functional.rms_norm(x, [3], weight=self.weight)

        x = torch.randn(2, 5, 3)

        onnx_program = self.export(RMSNormWithWeight(), (x,), opset_version=23)

        onnx_testing.assert_onnx_program(onnx_program)

    def test_rms_norm_with_eps(self):
        # [FUNCTION DOC] Exports RMS normalization with a custom epsilon at opset 23 and
        # checks numerical agreement.
        """Test RMS normalization with custom epsilon."""

        class RMSNormWithEps(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying RMS normalization with a custom epsilon.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns RMS normalization of the input over the
                # last dimension with epsilon of one in a hundred thousand.
                return torch.nn.functional.rms_norm(x, [3], eps=1e-5)

        x = torch.randn(2, 5, 3)

        onnx_program = self.export(RMSNormWithEps(), (x,), opset_version=23)

        onnx_testing.assert_onnx_program(onnx_program)

    def test_enable_gqa_in_attention_23_with_dropout(self):
        # [FUNCTION DOC] Exports grouped-query attention with dropout at opset 23. Because
        # dropout makes the output random, it only checks the graph contains the Unsqueeze,
        # Expand and Reshape nodes of the manual grouped-query path.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model computing scaled dot product attention with
            # grouped-query mode enabled and dropout probability of 0.1.
            def forward(self, q, k, v):
                # [FUNCTION DOC] HELPER: Returns scaled dot product attention with
                # grouped-query mode and dropout enabled.
                return torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                    q, k, v, enable_gqa=True, dropout_p=0.1
                )

        model = Model()

        query = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 2, 8, 16)
        value = torch.randn(2, 2, 8, 16)

        onnx_program = self.export(
            model,
            (
                query,
                key,
                value,
            ),
            opset_version=23,
        )
        # opset23 only uses manually gqa path when dropout is enabled,
        # and dropout makes the output non-deterministic,
        # so we check for the presence of the ops used in that path.
        all_ops = [node.op_type for node in onnx_program.model.graph]
        self.assertIn("Unsqueeze", all_ops)
        self.assertIn("Expand", all_ops)
        self.assertIn("Reshape", all_ops)

    def test_onnx_export_invoke_subgraph(self):
        # [FUNCTION DOC] Exports a model that calls a function wrapped as a nested compile
        # region, without optimization, and checks numerical agreement. Function
        # preservation in the graph is not yet checked (a TODO note).
        class InvokeSubgraphModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model calling an inner function through the nested
            # compile region wrapper.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Wraps the inner function as a nested compile
                # region, calls it with both inputs and returns its result.
                def inner_fn(a, b):
                    # [FUNCTION DOC] HELPER: Returns the product of the two arguments
                    # plus the first.
                    return torch.mul(a, b) + a

                return torch.compiler.nested_compile_region(inner_fn)(x, y)

        x = torch.randn(8)
        y = torch.randn(8)

        onnx_program = self.export(InvokeSubgraphModel(), (x, y), optimize=False)

        # TODO(justinchuby): Function preservation not implemented yet in ONNX exporter
        # # Verify that the function is preserved in the ONNX graph
        # # The function should appear in the model's functions list
        # onnx_model = onnx_program.model
        # self.assertGreater(
        #     len(onnx_model.functions),
        #     0,
        #     "Expected at least one function in the ONNX model",
        # )

        # Verify the output is correct
        onnx_testing.assert_onnx_program(onnx_program, args=(x, y))

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_onnx_export_invoke_subgraph_with_lifted_tensor_constant(self):
        # [FUNCTION DOC] Regression test, run with single-use invoke subgraph inlining
        # turned off. Exports a model calling the same nested compile region three times,
        # which captures a tensor constant, without optimization. Checks that no ONNX
        # function in the exported model uses an input that is a root graph initializer
        # unless it is one of the function's own inputs or an earlier node output; otherwise
        # fails the test.
        # Covers the full decoder-style pipeline: a region wrapped in
        # ``nested_compile_region`` that captures an inline tensor constant,
        # repeated several times so the region is materialized as an
        # invoke_subgraph HOP with a shared ``repeated_subgraph0`` submodule.
        # Exercises the decomposition / verifier / named_buffers /
        # FunctionalTensor / ONNX initializer paths end-to-end.
        @torch.compiler.nested_compile_region
        def block(x):
            # [FUNCTION DOC] HELPER: Nested compile region body; creates a constant
            # four-element tensor and returns the input times it plus one.
            w = torch.tensor([1.0, 2.0, 3.0, 4.0])
            return x * w + 1.0

        class RepeatedBlockModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying the compile region block three
            # times in sequence.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Passes the input through the block three times
                # and returns it.
                x = block(x)
                x = block(x)
                x = block(x)
                return x

        x = torch.randn(4)

        onnx_program = self.export(RepeatedBlockModel(), (x,), optimize=False)

        # Regardless of whether the subgraph is preserved as an ONNX function
        # or inlined by the version converter, the lifted tensor constant must
        # be self-contained in the exported model: no ONNX function should
        # silently reference a root-graph initializer from its outer scope
        # (which is invalid ONNX and was the bug being fixed).
        root_initializer_names = {
            initializer.name
            for initializer in onnx_program.model.graph.initializers.values()
        }
        for function in onnx_program.model.functions.values():
            function_value_names = {value.name for value in function.inputs}
            for node in function:
                for input_value in node.inputs:
                    if input_value is None:
                        continue
                    if (
                        input_value.name in root_initializer_names
                        and input_value.name not in function_value_names
                    ):
                        self.fail(
                            f"ONNX function {function.name} captures root graph "
                            f"initializer {input_value.name}"
                        )
                function_value_names.update(
                    output.name for output in node.outputs if output is not None
                )


if __name__ == "__main__":
    common_utils.run_tests()
