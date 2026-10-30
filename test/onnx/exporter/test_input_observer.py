# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests for InputObserver, the tool that wraps a model method to record the first
# few calls (inputs, outputs, latency) and then infers dynamic shapes and export arguments
# from them. Covers positional, keyword, optional, nested (list, dict, custom pytree class)
# and variadic inputs, batch dimension forcing, value_if_missing substitution, error
# reporting, export of the inferred result and discrepancy checking against ONNX. Imports
# pandas at import time; some tests run real torch.export and ONNX exports.

import itertools

import pandas

import torch
from torch.onnx import InputObserver
from torch.onnx._internal.exporter._input_observer import _infer_dynamic_dimensions
from torch.testing._internal import common_utils


class TestInputObserver(common_utils.TestCase):
    # [CLASS DOC] Test case holding all InputObserver tests. Stateless; each test defines a
    # tiny model, drives it with several input sets under the observer (which stores at most
    # three calls unless told otherwise) and checks the stored count, outputs, inferred
    # dynamic shapes and inferred arguments.
    def test_infer_dynamic_dimensions(self):
        # [FUNCTION DOC] Checks the dimension inference helper on lists of shapes: only
        # dimension indices whose sizes differ between shapes are returned (index two for one
        # list, indices zero and two for the other).
        self.assertEqual([2], _infer_dynamic_dimensions([(1, 2, 3), (1, 2, 4)]))
        self.assertEqual([0, 2], _infer_dynamic_dimensions([(1, 2, 3), (2, 2, 4)]))

    def test_io_captured_args(self):
        # [FUNCTION DOC] Observes a two-input model called four times with positional tensors.
        # Asserts only three calls are stored, each with one output equal to the eager result,
        # that the inferred shapes are a tuple marking both dims of the first input and dim
        # one of the second as dynamic, and that inferred arguments are a tuple of two.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the sum of its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((7, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({0: cst, 1: cst}, {1: cst}), observer.infer_dynamic_shapes())
        args = observer.infer_arguments()
        self.assertIsInstance(args, tuple)
        self.assertEqual(2, len(args))

    def test_io_captured_not_forward(self):
        # [FUNCTION DOC] Observes a method named notforward instead of forward by passing the
        # method name. Four calls through forward run it, three are stored with one output
        # each. Asserts the inferred shapes tuple and arguments tuple concern only the single
        # argument of that method (dim one dynamic).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a helper method that returns the absolute value
            # of its input; forward adds the first input and the helper applied to the second
            # input.
            def notforward(self, w):
                # [FUNCTION DOC] HELPER: Returns the absolute value of w.
                return w.abs()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the absolute value of y computed
                # through the helper method.
                return x + self.notforward(y)

        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((7, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model, method_name="notforward"):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({1: cst},), observer.infer_dynamic_shapes())
        args = observer.infer_arguments()
        self.assertIsInstance(args, tuple)
        self.assertEqual(1, len(args))

    def test_io_captured_kwargs(self):
        # [FUNCTION DOC] Same as the positional case but every call passes keyword arguments.
        # Asserts three stored calls with correct outputs, inferred shapes as a mapping keyed
        # by argument name, and inferred arguments as a mapping of two entries.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the sum of its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8))),
            dict(x=torch.randn((7, 9)), y=torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}), observer.infer_dynamic_shapes()
        )
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(2, len(args))

    def test_io_captured_kwargs_bool(self):
        # [FUNCTION DOC] Calls a model with an extra boolean keyword always set to False,
        # which differs from its default True. Asserts the dynamic shapes mapping holds None
        # for that boolean and the inferred arguments mapping has three entries, so
        # non-default scalar keywords are kept as constants.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the sum of x and y when its boolean keyword
            # is true, otherwise their difference.
            def forward(self, x, y, add=True):
                # [FUNCTION DOC] HELPER: Returns x plus y if add is true, otherwise x minus y.
                if add:
                    return x + y
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6)), add=False),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7)), add=False),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8)), add=False),
            dict(x=torch.randn((7, 9)), y=torch.randn((1, 9)), add=False),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, add=None),
            observer.infer_dynamic_shapes(),
        )
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(3, len(args))

    def test_io_captured_args_kwargs(self):
        # [FUNCTION DOC] Calls a model with two positional tensors and two tensor keywords.
        # Asserts three stored calls with correct outputs, shapes as a mapping over all four
        # names, and four inferred arguments in a mapping.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model summing x and y plus the optional z and w tensors when
            # they are given.
            def forward(self, x, y, z=None, w=None):
                # [FUNCTION DOC] HELPER: Returns x plus y, then adds z and w in place when
                # they are not None.
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(z=torch.randn((5, 6)), w=torch.randn((1, 6))),
            ),
            (
                (torch.randn((6, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((6, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((7, 8)), torch.randn((1, 8))),
                dict(z=torch.randn((7, 8)), w=torch.randn((1, 8))),
            ),
            (
                (torch.randn((8, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((8, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(),
        )
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(4, len(args))

    def test_io_captured_optional_args(self):
        # [FUNCTION DOC] Observes calls where the optional second argument is absent in the
        # first call. Asserts outputs match eager and the inferred shapes are a tuple of two
        # entries, dims zero and one for the first and dim one for the second.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning x when y is not given, otherwise x minus y.
            def forward(self, x, y=None):
                # [FUNCTION DOC] HELPER: Returns x if y is None, otherwise x minus y.
                if y is None:
                    return x
                return x - y

        inputs = [
            (torch.randn((5, 6)),),
            (torch.randn((6, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((8, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({0: cst, 1: cst}, {1: cst}), observer.infer_dynamic_shapes())

    def test_infer_arguments_optional(self):
        # [FUNCTION DOC] Same optional second argument scenario, then asks for arguments of
        # the first recorded call by index. Asserts a tuple of two tensors where the missing
        # one is replaced by an empty tensor of shape one by zero.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning x when y is not given, otherwise x minus y.
            def forward(self, x, y=None):
                # [FUNCTION DOC] HELPER: Returns x if y is None, otherwise x minus y.
                if y is None:
                    return x
                return x - y

        inputs = [
            (torch.randn((5, 6)),),
            (torch.randn((6, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((8, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({0: cst, 1: cst}, {1: cst}), observer.infer_dynamic_shapes())
        infer_args = observer.infer_arguments(0)
        self.assertIsInstance(infer_args, tuple)
        self.assertEqual(len(infer_args), 2)
        self.assertIsInstance(infer_args[0], torch.Tensor)
        self.assertIsInstance(infer_args[1], torch.Tensor)
        self.assertEqual(infer_args[0].shape, (5, 6))
        self.assertEqual(infer_args[1].shape, (1, 0))

    def test_io_captured_optional_kwargs(self):
        # [FUNCTION DOC] Observes keyword calls in which the optional y keyword is missing in
        # the first call. Asserts outputs match eager and the shapes mapping covers both
        # names.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning x when y is not given, otherwise x minus y.
            def forward(self, x, y=None):
                # [FUNCTION DOC] HELPER: Returns x if y is None, otherwise x minus y.
                if y is None:
                    return x
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6))),
            dict(x=torch.randn((6, 7)), y=torch.randn((1, 7))),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8))),
            dict(x=torch.randn((8, 9)), y=torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}), observer.infer_dynamic_shapes()
        )

    def test_io_captured_optional_args_kwargs(self):
        # [FUNCTION DOC] Observes calls mixing positional and keyword tensors where some
        # optional ones are absent in the first call. Asserts outputs match eager and the
        # shapes mapping holds an entry for x, y, z and w.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model summing x with the optional y, z and w tensors when
            # given.
            def forward(self, x, y=None, z=None, w=None):
                # [FUNCTION DOC] HELPER: Returns x plus y if y is given, then adds z and w
                # when they are not None.
                r = x + y if y is not None else x
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)),),
                dict(w=torch.randn((1, 6))),
            ),
            (
                (torch.randn((6, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((6, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((7, 8)), torch.randn((1, 8))),
                dict(z=torch.randn((7, 8)), w=torch.randn((1, 8))),
            ),
            (
                (torch.randn((8, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((8, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(),
        )

    def test_io_captured_not_supported_kwargs(self):
        # [FUNCTION DOC] Observes calls where the first passes only x and the others pass only
        # y, so no recorded call contains all arguments. Asserts that inferring dynamic shapes
        # raises a RuntimeError saying at least one call to the observed model must have all
        # arguments.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model of two optional inputs returning whichever is present,
            # or their difference when both are.
            def forward(self, x=None, y=None):
                # [FUNCTION DOC] HELPER: Returns x if y is None, y if x is None, else x minus
                # y.
                if y is None:
                    return x
                if x is None:
                    return y
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6))),
            dict(y=torch.randn((1, 7))),
            dict(y=torch.randn((1, 7))),
            dict(y=torch.randn((1, 7))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        with self.assertRaisesRegex(
            RuntimeError, "At least one call to the observed model"
        ):
            observer.infer_dynamic_shapes()

    def test_io_captured_incompatible_number_of_flattened_kwargs(self):
        # [FUNCTION DOC] Observes keyword calls where the list in y has a different number of
        # tensors across calls. Asserts inferring shapes raises a RuntimeError whose message
        # starts with the named argument y having inconsistent flattened tensor counts.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with optional x and a list-valued y; returns x, the
            # first list element, or their difference.
            def forward(self, x=None, y=None):
                # [FUNCTION DOC] HELPER: Returns x if y is None, y first element if x is None,
                # else x minus the first element of y.
                if y is None:
                    return x
                if x is None:
                    return y[0]
                return x - y[0]

        inputs = [
            dict(x=torch.randn((5, 6))),
            dict(x=torch.randn((5, 7)), y=[torch.randn((1, 7))]),
            dict(x=torch.randn((5, 7)), y=[torch.randn((1, 7)), torch.randn((1, 7))]),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        with self.assertRaisesRegex(RuntimeError, "Named argument 'y' has"):
            observer.infer_dynamic_shapes()

    def test_io_captured_incompatible_number_of_flattened_args(self):
        # [FUNCTION DOC] Asserts inferring shapes before anything was observed raises a
        # RuntimeError saying no inputs were captured. Then observes two calls whose list
        # argument has different lengths and asserts a RuntimeError reporting the positional
        # argument at index one has inconsistent counts.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model subtracting the first element of a list argument from
            # x.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x minus the first element of y.
                return x - y[0]

        inputs = [
            (torch.randn((5, 7)), [torch.randn((1, 7))]),
            (torch.randn((5, 7)), [torch.randn((1, 7)), torch.randn((1, 7))]),
        ]

        model = Model()
        observer = InputObserver()
        with self.assertRaisesRegex(RuntimeError, "No inputs were captured."):
            observer.infer_dynamic_shapes()
        with observer(model):
            for args in inputs:
                model(*args)
        with self.assertRaisesRegex(RuntimeError, "Positional argument 1 has"):
            observer.infer_dynamic_shapes()

    def test_io_captured_args_list(self):
        # [FUNCTION DOC] Observes calls with a tensor and a list of two tensors. Asserts three
        # stored calls and outputs, and that the shapes are a tuple whose second entry is a
        # list of two mappings with dim one dynamic.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding x and the two elements of a list argument.
            def forward(self, x, y_list):
                # [FUNCTION DOC] HELPER: Returns x plus the first and second elements of
                # y_list.
                return x + y_list[0] + y_list[1]

        inputs = [
            (torch.randn((5, 6)), [torch.randn((1, 6)), torch.randn((1, 6))]),
            (torch.randn((7, 7)), [torch.randn((1, 7)), torch.randn((1, 7))]),
            (torch.randn((7, 8)), [torch.randn((1, 8)), torch.randn((1, 8))]),
            (torch.randn((7, 9)), [torch.randn((1, 9)), torch.randn((1, 9))]),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            ({0: cst, 1: cst}, [{1: cst}, {1: cst}]), observer.infer_dynamic_shapes()
        )

    def test_io_captured_args_list_list(self):
        # [FUNCTION DOC] Like the list case but the second list element is itself a
        # one-element list. Asserts the inferred shapes keep that nesting.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding x, the first list element and the first element
            # of the nested list.
            def forward(self, x, y_list):
                # [FUNCTION DOC] HELPER: Returns x plus y_list first element plus the first
                # element of the nested second element.
                return x + y_list[0] + y_list[1][0]

        inputs = [
            (torch.randn((5, 6)), [torch.randn((1, 6)), [torch.randn((1, 6))]]),
            (torch.randn((7, 7)), [torch.randn((1, 7)), [torch.randn((1, 7))]]),
            (torch.randn((7, 8)), [torch.randn((1, 8)), [torch.randn((1, 8))]]),
            (torch.randn((7, 9)), [torch.randn((1, 9)), [torch.randn((1, 9))]]),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            ({0: cst, 1: cst}, [{1: cst}, [{1: cst}]]), observer.infer_dynamic_shapes()
        )

    def test_io_captured_args_dict(self):
        # [FUNCTION DOC] Observes calls with a tensor and a mapping argument. First checks the
        # expected shapes are accepted by torch.export on the last input, then asserts the
        # observer infers exactly those shapes (a nested mapping with one entry per key) and
        # stored three calls.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding x and the values under keys x and y of a
            # mapping argument.
            def forward(self, x, y_dict):
                # [FUNCTION DOC] HELPER: Returns x plus the x entry plus the y entry of
                # y_dict.
                return x + y_dict["x"] + y_dict["y"]

        inputs = [
            (torch.randn((5, 6)), dict(x=torch.randn((1, 6)), y=torch.randn((1, 6)))),
            (torch.randn((7, 7)), dict(x=torch.randn((1, 7)), y=torch.randn((1, 7)))),
            (torch.randn((7, 8)), dict(x=torch.randn((1, 8)), y=torch.randn((1, 8)))),
            (torch.randn((7, 9)), dict(x=torch.randn((1, 9)), y=torch.randn((1, 9)))),
        ]

        cst = torch.export.Dim.DYNAMIC
        expected = ({0: cst, 1: cst}, dict(x={1: cst}, y={1: cst}))
        model = Model()
        torch.export.export(model, inputs[-1], dynamic_shapes=expected)

        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)

        self.assertEqual(
            ({0: cst, 1: cst}, dict(x={1: cst}, y={1: cst})),
            observer.infer_dynamic_shapes(),
        )

    def test_io_captured_args_dict_args_kwargs(self):
        # [FUNCTION DOC] Observes three calls where the first lacks the optional tuple keyword
        # and later ones pass it. Verifies the expected shapes are accepted by torch.export on
        # the last call, then asserts three stored calls and that the inferred shapes equal
        # them (mapping with a list entry and a tuple entry).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding x and a list of two tensors, and optionally a
            # tuple of two tensors.
            def forward(self, x, y_list, z_tuple=None):
                # [FUNCTION DOC] HELPER: Sums x and both list elements, plus both tuple
                # elements when the tuple is given.
                if z_tuple is None:
                    return x + y_list[0] + y_list[1]
                return x + y_list[0] + y_list[1] + z_tuple[0] + z_tuple[1]

        inputs = [
            ((torch.randn((5, 6)), [torch.randn((5, 6)), torch.randn((1, 6))]), {}),
            (
                (torch.randn((6, 7)), [torch.randn((6, 7)), torch.randn((1, 7))]),
                {"z_tuple": (torch.randn((6, 7)), torch.randn((1, 7)))},
            ),
            (
                (torch.randn((7, 8)), [torch.randn((7, 8)), torch.randn((1, 8))]),
                {"z_tuple": (torch.randn((7, 8)), torch.randn((1, 8)))},
            ),
        ]

        cst = torch.export.Dim.DYNAMIC
        expected = dict(
            x={0: cst, 1: cst},
            y_list=[{0: cst, 1: cst}, {1: cst}],
            z_tuple=({0: cst, 1: cst}, {1: cst}),
        )
        model = Model()
        torch.export.export(
            model, inputs[-1][0], kwargs=inputs[-1][1], dynamic_shapes=expected
        )

        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
        self.assertEqual(expected, observer.infer_dynamic_shapes())

    def test_io_captured_custom_class(self):
        # [FUNCTION DOC] Registers a custom class with the pytree machinery (global side
        # effect, with a fixed serialized name), observes calls where the optional custom
        # argument is absent in the first call, and asserts the inferred shapes flatten the
        # custom object into one entry per contained tensor. Also checks the sample flattens
        # to five leaves, runs the model and verifies torch.export accepts the expected
        # shapes.
        class TestCustomClass:
            # [CLASS DOC] HELPER: Container of key and value pairs built from two parallel
            # lists; stores a list of pairs in its data attribute.
            def __init__(self, keys, values):
                # [FUNCTION DOC] HELPER: Pairs the keys with the values and stores them as the
                # data list.
                self.data = list(zip(keys, values))

        def _flatten(custom):
            # [FUNCTION DOC] HELPER: Pytree flatten function; returns the pair elements as one
            # flat list and a context list of generated key and value names.
            data = custom.data
            flat = list(itertools.chain.from_iterable(data))
            keys = list(
                itertools.chain.from_iterable(
                    (f"key_{i}", f"value_{i}") for i in range(len(data))
                )
            )
            return flat, keys

        def _flatten_with_keys(custom):
            # [FUNCTION DOC] HELPER: Pytree flatten with keys function; returns key and value
            # pairs using mapping keys built from the context names, plus the context.
            values, context = _flatten(custom)
            return [
                (torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)
            ], context

        def _unflatten(values, context, output_type=None):
            # [FUNCTION DOC] HELPER: Pytree unflatten function; rebuilds the container taking
            # even positions as keys and odd positions as values. The output type argument is
            # ignored.
            return TestCustomClass(values[::2], values[1::2])

        torch.utils._pytree.register_pytree_node(
            TestCustomClass,
            _flatten,
            _unflatten,
            serialized_type_name="onnxtest.TestCustomClass",
            flatten_with_keys_fn=_flatten_with_keys,
        )

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning x when the custom container is missing or
            # empty, else x plus the four tensors it holds.
            def forward(self, x, custom=None):
                # [FUNCTION DOC] HELPER: Returns x, or x plus the four stored tensors if
                # custom is given.
                if not custom:
                    return x
                data = custom.data
                return x + data[0][0] + data[0][1] + data[1][0] + data[1][1]

        inputs = [
            (torch.randn((5, 6)),),
            (
                torch.randn((6, 7)),
                TestCustomClass(
                    [torch.randn((6, 7)), torch.randn((1, 7))],
                    [torch.randn((1, 7)), torch.randn((6, 7))],
                ),
            ),
            (
                torch.randn((7, 8)),
                TestCustomClass(
                    [torch.randn((7, 8)), torch.randn((1, 8))],
                    [torch.randn((1, 8)), torch.randn((7, 8))],
                ),
            ),
        ]

        cst = torch.export.Dim.DYNAMIC
        expected = (
            {0: cst, 1: cst},
            [{0: cst, 1: cst}, {1: cst}, {1: cst}, {0: cst, 1: cst}],
        )
        flat = torch.utils._pytree.tree_flatten(inputs[-1])[0]
        self.assertEqual(len(flat), 5)

        model = Model()
        model(*inputs[-1])
        torch.export.export(model, inputs[-1], dynamic_shapes=expected)
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(expected, observer.infer_dynamic_shapes())

    def test_io_captured_args_kwargs_dynamic_batch(self):
        # [FUNCTION DOC] Observes calls where the first dimension never changes. Asserts that
        # asking for a dynamic batch dimension on selected inputs, given either by position
        # and name or by names, marks dim zero dynamic for those inputs; the expected shapes
        # mapping is the same for both requests.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model summing x and y plus the optional z and w tensors when
            # given.
            def forward(self, x, y, z=None, w=None):
                # [FUNCTION DOC] HELPER: Returns x plus y, then adds z and w in place when
                # they are not None.
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(z=torch.randn((5, 6)), w=torch.randn((1, 6))),
            ),
            (
                (torch.randn((5, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((5, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((5, 8)), torch.randn((1, 8))),
                dict(z=torch.randn((5, 8)), w=torch.randn((1, 8))),
            ),
            (
                (torch.randn((5, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((5, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(set_batch_dimension_for={0, "z"}),
        )
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(set_batch_dimension_for={"x", "z"}),
        )

    def test_io_captured_different_order(self):
        # [FUNCTION DOC] Observes calls whose keyword arguments come in alternating order.
        # Asserts the inferred shapes are unaffected by the order and by how batch dimension
        # inputs are named. Then exports to ONNX with the inferred arguments and shapes, runs
        # the discrepancy check without a progress bar, and asserts the largest absolute error
        # is below one in a hundred thousand.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model summing x and y plus the optional z and w tensors when
            # given.
            def forward(self, x, y, z=None, w=None):
                # [FUNCTION DOC] HELPER: Returns x plus y, then adds z and w in place when
                # they are not None.
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(w=torch.randn((1, 6)), z=torch.randn((5, 6))),
            ),
            (
                (torch.randn((5, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((5, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((5, 8)), torch.randn((1, 8))),
                dict(w=torch.randn((1, 8)), z=torch.randn((5, 8))),
            ),
            (
                (torch.randn((5, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((5, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(set_batch_dimension_for={0, "z"}),
        )
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(set_batch_dimension_for={"x", "z"}),
        )
        epo = torch.onnx.export(
            model,
            (),
            kwargs=observer.infer_arguments(),
            dynamic_shapes=observer.infer_dynamic_shapes(set_batch_dimension_for=True),
        )
        data = observer.check_discrepancies(epo, progress_bar=False)
        df = pandas.DataFrame(data)
        self.assertLess(df["abs"].max(), 1e-5)

    def test_io_check_discrepancies(self):
        # [FUNCTION DOC] Observes four positional calls (three stored), exports to ONNX with
        # inferred arguments and shapes with batch dimension forcing, and checks
        # discrepancies. Asserts one mapping per stored call and that the absolute errors are
        # small, also via a data frame.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the sum of its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((7, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)

        epo = torch.onnx.export(
            model,
            observer.infer_arguments(),
            dynamic_shapes=observer.infer_dynamic_shapes(set_batch_dimension_for=True),
        )
        data = observer.check_discrepancies(epo, progress_bar=False)
        self.assertEqual(len(data), 3)
        self.assertIsInstance(data[0], dict)
        self.assertLess(max(obs["abs"] for obs in data), 1e-5)
        df = pandas.DataFrame(data)
        self.assertLess(df["abs"].max(), 1e-5)

    def test_io_infer_arguments(self):
        # [FUNCTION DOC] Observes calls with the z and w keywords in alternating order, then
        # asks for arguments completed from a user supplied mapping and from a user supplied
        # tuple. Asserts four entries come back each time, with the given tensors kept and the
        # missing ones replaced by empty tensors with a zero-sized last dimension.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model summing x and y plus the optional z and w tensors when
            # given.
            def forward(self, x, y, z=None, w=None):
                # [FUNCTION DOC] HELPER: Returns x plus y, then adds z and w in place when
                # they are not None.
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(w=torch.randn((1, 6)), z=torch.randn((5, 6))),
            ),
            (
                (torch.randn((5, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((5, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((5, 8)), torch.randn((1, 8))),
                dict(w=torch.randn((1, 8)), z=torch.randn((5, 8))),
            ),
            (
                (torch.randn((5, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((5, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        iargs = observer.infer_arguments(
            dict(w=torch.randn((1, 6)), z=torch.randn((5, 6)))
        )
        self.assertEqual(len(iargs), 4)
        self.assertEqual(iargs["x"].shape, (5, 0))
        self.assertEqual(iargs["y"].shape, (1, 0))
        self.assertEqual(iargs["w"].shape, (1, 6))
        self.assertEqual(iargs["z"].shape, (5, 6))

        iargs = observer.infer_arguments((torch.randn((5, 6)), torch.randn((1, 6))))
        self.assertEqual(len(iargs), 4)
        self.assertEqual(iargs["x"].shape, (5, 6))
        self.assertEqual(iargs["y"].shape, (1, 6))
        self.assertEqual(iargs["w"].shape, (1, 0))
        self.assertEqual(iargs["z"].shape, (5, 0))

    def test_io_mixed_args_kwargs_as_dict_1(self):
        # [FUNCTION DOC] Observes four calls alternating between positional-only and
        # keyword-only styles with the number of stored calls raised to four. Asserts all four
        # stored, that each recorded candidate keeps as many flat tensors as aligned non-empty
        # ones, that shapes are a mapping with dim one dynamic for x and y, and that two
        # non-None arguments are inferred as a mapping.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model of two optional inputs returning x alone or the sum of
            # x and y.
            def forward(self, x=None, y=None):
                # [FUNCTION DOC] HELPER: Returns x if y is None, else x plus y.
                if y is None:
                    return x
                return x + y

        inputs = [
            ((torch.randn((5, 6)),), dict()),
            ((), dict(x=torch.randn((5, 7)), y=torch.randn((5, 7)))),
            ((torch.randn((5, 8)),), dict()),
            ((), dict(x=torch.randn((5, 9)), y=torch.randn((5, 9)))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model, store_n_calls=4):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 4)
        observer.infer_dynamic_shapes()
        for cand in observer.info.inputs:
            cand.str_obs()
            self.assertEqual(
                len(cand.flat_list),
                len([t for t in cand.aligned_flat_list if t is not None]),
            )

        cst = torch.export.Dim.DYNAMIC
        dynamic_shapes = observer.infer_dynamic_shapes()
        self.assertEqual({"x": {1: cst}, "y": {1: cst}}, dynamic_shapes)
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(2, len(args))
        self.assertEqual(len([v for v in args.values() if v is not None]), 2)

    def test_io_int_kwargs(self):
        # [FUNCTION DOC] Calls a model with an integer keyword differing from its default.
        # Asserts the inferred arguments keep it as a constant, the shapes mapping holds None
        # for it, torch.export works and matches eager, and the exported ONNX graph has only
        # the two tensor inputs.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning x plus y when the integer option is one,
            # else x minus y.
            def forward(self, x=None, y=None, option=1):
                # [FUNCTION DOC] HELPER: Returns x plus y if option equals one, else x minus
                # y.
                if option == 1:
                    return x + y
                return x - y

        inputs = [
            dict(x=torch.randn((5, 7)), y=torch.randn((5, 7)), option=0),
            dict(x=torch.randn((5, 9)), y=torch.randn((5, 9)), option=0),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model, store_n_calls=4):
            for kwargs in inputs:
                model(**kwargs)
        kwargs = observer.infer_arguments()
        self.assertIn("option", kwargs)
        self.assertEqual(kwargs["option"], 0)
        shapes = observer.infer_dynamic_shapes()
        self.assertIn("option", shapes)
        self.assertEqual(shapes["option"], None)
        ep = torch.export.export(model, (), kwargs=kwargs, dynamic_shapes=shapes)
        torch.testing.assert_close(model(**kwargs), ep.module()(**kwargs))
        epo = torch.onnx.export(model, (), kwargs=kwargs, dynamic_shapes=shapes)
        proto = epo.model_proto
        self.assertEqual(["x", "y"], [i.name for i in proto.graph.input])

    def test_io_mixed_args_kwargs_as_dict_2(self):
        # [FUNCTION DOC] Observes four calls alternating between only y and both x and y,
        # storing all four. Asserts that inferring dynamic shapes raises a RuntimeError.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning y when x is missing, else x plus y.
            def forward(self, x=None, y=None):
                # [FUNCTION DOC] HELPER: Returns y if x is None, else x plus y.
                if x is None:
                    return y
                return x + y

        inputs = [
            ((), dict(y=torch.randn((5, 6)))),
            ((torch.randn((5, 7)), torch.randn((5, 7))), dict()),
            ((), dict(y=torch.randn((5, 8)))),
            ((torch.randn((5, 9)), torch.randn((5, 9))), dict()),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model, store_n_calls=4):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 4)
        with self.assertRaises(RuntimeError):
            observer.infer_dynamic_shapes()

    def test_infer_dynamic_shapes_missing(self):
        # [FUNCTION DOC] Observes a multimodal-like model where the first call has image
        # pixels and no cache and later calls have a cache but no pixels. The observer is
        # given a replacement value for the pixels. Asserts batch forced shapes per argument
        # name, that argument order is preserved, and that the inferred pixel tensor is empty
        # with the given shape.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with seven optional multimodal-style arguments that
            # returns only the token ids.
            def forward(
                self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                token_type_ids=None,
                cache_position=None,
            ):
                # [FUNCTION DOC] HELPER: Returns input_ids unchanged; other arguments are
                # ignored.
                return input_ids

        inputs = [
            dict(
                input_ids=torch.ones((1, 28), dtype=torch.int64),
                pixel_values=torch.ones((1, 3, 112, 112), dtype=torch.int64),
                attention_mask=torch.ones((1, 28), dtype=torch.int64),
                position_ids=torch.ones((1, 28), dtype=torch.int64),
                token_type_ids=torch.ones((1, 28), dtype=torch.int64),
                cache_position=torch.ones((28,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 29), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=torch.rand((1, 1, 28, 32)),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 30), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=torch.rand((1, 1, 29, 32)),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
        ]

        model = Model()
        observer = InputObserver(
            value_if_missing=dict(pixel_values=torch.empty((0, 3, 112, 112)))
        )
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)

        shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
        cst = torch.export.Dim.DYNAMIC
        expected = {
            "input_ids": {0: cst, 1: cst},
            "pixel_values": {0: cst},
            "attention_mask": {0: cst, 1: cst},
            "position_ids": {0: cst, 1: cst},
            "past_key_values": {0: cst, 2: cst},
            "token_type_ids": {0: cst, 1: cst},
            "cache_position": {0: cst},
        }
        self.assertEqual(expected, shapes)
        kwargs = observer.infer_arguments()
        self.assertEqual(list(expected), list(kwargs))
        self.assertEqual((0, 3, 112, 112), kwargs["pixel_values"].shape)

    def test_infer_dynamic_shapes_missing_args(self):
        # [FUNCTION DOC] Same multimodal scenario with positional arguments and a replacement
        # value given by argument position. Asserts the shapes tuple has four entries, the
        # inferred arguments have four entries, and the second argument is the empty tensor.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with four optional multimodal-style arguments that
            # returns only the token ids.
            def forward(
                self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                past_key_values=None,
            ):
                # [FUNCTION DOC] HELPER: Returns input_ids unchanged; other arguments are
                # ignored.
                return input_ids

        inputs = [
            (
                torch.ones((1, 28), dtype=torch.int64),
                torch.ones((1, 3, 112, 112), dtype=torch.int64),
                torch.ones((1, 28), dtype=torch.int64),
            ),
            (
                torch.ones((1, 1), dtype=torch.int64),
                None,
                torch.ones((1, 29), dtype=torch.int64),
                torch.rand((1, 1, 28, 32)),
            ),
            (
                torch.ones((1, 1), dtype=torch.int64),
                None,
                torch.ones((1, 30), dtype=torch.int64),
                torch.rand((1, 1, 29, 32)),
            ),
        ]

        model = Model()
        observer = InputObserver(
            value_if_missing={1: torch.empty((0, 3, 112, 112), dtype=torch.int64)}
        )
        with observer(model):
            for args in inputs:
                model(*args)

        shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
        cst = torch.export.Dim.DYNAMIC
        expected = ({0: cst, 1: cst}, {0: cst}, {0: cst, 1: cst}, {0: cst, 2: cst})
        self.assertEqual(expected, shapes)
        args = observer.infer_arguments()
        self.assertEqual(len(expected), len(args))
        self.assertEqual((0, 3, 112, 112), args[1].shape)

    def test_infer_dynamic_shapes_missing_kwargs_nested(self):
        # [FUNCTION DOC] Multimodal scenario where the replacement value for the missing
        # pixels is a tuple of two tensors. Asserts shapes keep the tuple structure with dim
        # zero dynamic for each element and the inferred pixels are a tuple of two empty
        # tensors with the given shape.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with seven optional multimodal-style arguments that
            # returns only the token ids.
            def forward(
                self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                token_type_ids=None,
                cache_position=None,
            ):
                # [FUNCTION DOC] HELPER: Returns input_ids unchanged; other arguments are
                # ignored.
                return input_ids

        inputs = [
            dict(
                input_ids=torch.ones((1, 28), dtype=torch.int64),
                pixel_values=(
                    torch.ones((1, 3, 112, 112), dtype=torch.int64),
                    torch.ones((1, 3, 112, 112), dtype=torch.int64),
                ),
                attention_mask=torch.ones((1, 28), dtype=torch.int64),
                position_ids=torch.ones((1, 28), dtype=torch.int64),
                token_type_ids=torch.ones((1, 28), dtype=torch.int64),
                cache_position=torch.ones((28,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 29), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=torch.rand((1, 1, 28, 32)),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 30), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=torch.rand((1, 1, 29, 32)),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
        ]

        model = Model()
        observer = InputObserver(
            value_if_missing=dict(
                pixel_values=(
                    torch.empty((0, 3, 112, 112), dtype=torch.int64),
                    torch.empty((0, 3, 112, 112), dtype=torch.int64),
                )
            )
        )
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)

        shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
        cst = torch.export.Dim.DYNAMIC
        expected = {
            "input_ids": {0: cst, 1: cst},
            "pixel_values": ({0: cst}, {0: cst}),
            "attention_mask": {0: cst, 1: cst},
            "position_ids": {0: cst, 1: cst},
            "past_key_values": {0: cst, 2: cst},
            "token_type_ids": {0: cst, 1: cst},
            "cache_position": {0: cst},
        }
        self.assertEqual(expected, shapes)
        kwargs = observer.infer_arguments()
        self.assertEqual(list(expected), list(kwargs))
        self.assertIsInstance(kwargs["pixel_values"], tuple)
        self.assertEqual(2, len(kwargs["pixel_values"]))
        self.assertEqual((0, 3, 112, 112), kwargs["pixel_values"][0].shape)
        self.assertEqual((0, 3, 112, 112), kwargs["pixel_values"][1].shape)

    def test_io_captured_kwargs_kwargs(self):
        # [FUNCTION DOC] Observes a model taking a variadic keyword mapping. Asserts the
        # inferred shapes nest the extra keyword under the variadic name, that inferred
        # arguments are a flat mapping of the names x and y, and that the torch.export
        # additional-inputs helper produces matching shapes (as tuples, None for the unchanged
        # dim).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model taking x plus any extra keywords and returning x plus
            # the extra keyword y.
            def forward(self, x, **kwargs):
                # [FUNCTION DOC] HELPER: Returns x plus the y entry of the extra keyword
                # mapping.
                return x + kwargs["y"]

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8))),
            dict(x=torch.randn((7, 9)), y=torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        ds = observer.infer_dynamic_shapes()
        self.assertEqual(dict(x={0: cst, 1: cst}, kwargs=dict(y={1: cst})), ds)
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(2, len(args))
        self.assertEqual(["x", "y"], list(args))

        dynamic_shapes = torch.export.AdditionalInputs()
        for kwargs in inputs:
            dynamic_shapes.add((), kwargs)
        dss = dynamic_shapes.dynamic_shapes(model, (), inputs[0])
        self.assertEqual({"x": (cst, cst), "kwargs": {"y": (None, cst)}}, dss)

    def test_io_captured_kwargs_kwargs_with_args(self):
        # [FUNCTION DOC] Observes a model taking a first argument, variadic positional
        # arguments and variadic keywords, with two calls. Asserts the nested shapes under the
        # names a, args and kwargs, compares with the torch.export additional-inputs helper,
        # asserts inferring arguments in the default form raises a RuntimeError, and that
        # asking for separate args and kwargs gives a keyword mapping with x and y and three
        # positional tensors.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model combining a first argument, two extra positional
            # tensors and two extra keyword tensors.
            def forward(self, a, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Returns a minus the product of the first two extra
                # positional tensors, plus the x keyword, minus the y keyword.
                return a - args[0] * args[1] + kwargs["x"] - kwargs["y"]

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((5, 6)), torch.randn((5, 6))),
                dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            ),
            (
                (torch.randn((7, 7)), torch.randn((7, 7)), torch.randn((7, 7))),
                dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 2)
        for i in range(2):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        ds = observer.infer_dynamic_shapes()
        self.assertEqual(
            {
                "a": {0: cst, 1: cst},
                "args": ({0: cst, 1: cst}, {0: cst, 1: cst}),
                "kwargs": {"x": {0: cst, 1: cst}, "y": {1: cst}},
            },
            ds,
        )

        dynamic_shapes = torch.export.AdditionalInputs()
        for args, kwargs in inputs:
            dynamic_shapes.add(args, kwargs)
        dss = dynamic_shapes.dynamic_shapes(model, *inputs[0])
        self.assertEqual(
            {
                "a": (cst, cst),
                "args": ((cst, cst), (cst, cst)),
                "kwargs": {"x": (cst, cst), "y": (None, cst)},
            },
            dss,
        )

        with self.assertRaises(RuntimeError):
            observer.infer_arguments()

        args, kwargs = observer.infer_arguments(as_args_kwargs=True)
        self.assertIsInstance(kwargs, dict)
        self.assertEqual(["x", "y"], list(kwargs))
        self.assertIsInstance(args, tuple)
        self.assertEqual(len(args), 3)


if __name__ == "__main__":
    common_utils.run_tests()
