# Owner(s): ["oncall: fx"]
# [MODULE DOC] Tests for the experimental dynamism tracker, which compares a list of example
# inputs and reports for every leaf path which tensor dimensions or scalar values vary
# across examples. Covers plain tensors and numbers, deeply nested containers, mixed content
# and module-valued entries. Running the file directly raises an error stating the tests are
# not currently used and must be enabled in the test discovery list.

import torch
from torch.fx.experimental._dynamism import track_dynamism_across_examples
from torch.testing._internal.common_utils import TestCase


class TestDynamism(TestCase):
    # [CLASS DOC] Tests for the dynamism tracking utility. Each test builds five or two
    # example mappings, calls the tracker and compares with an expected mapping from key to
    # path strings to tuples of booleans (true where the dimension or value differs across
    # examples).
    def test_dynamic_tensor(self):
        # [FUNCTION DOC] Five examples with a growing integer, a matrix whose first
        # dimension grows and a tensor nested inside a dictionary whose only dimension
        # grows. Expects the scalar and growing dimensions flagged true and the fixed second
        # matrix dimension false.
        ex1 = {"x": 1, "y": torch.ones(1, 1), "z": {0: torch.ones(1)}}
        ex2 = {"x": 2, "y": torch.ones(2, 1), "z": {0: torch.ones(2)}}
        ex3 = {"x": 3, "y": torch.ones(3, 1), "z": {0: torch.ones(3)}}
        ex4 = {"x": 4, "y": torch.ones(4, 1), "z": {0: torch.ones(4)}}
        ex5 = {"x": 5, "y": torch.ones(5, 1), "z": {0: torch.ones(5)}}
        examples = [ex1, ex2, ex3, ex4, ex5]

        result = track_dynamism_across_examples(examples)
        expected = {
            "x": {"L['x']": (True,)},
            "y": {"L['y']": (True, False)},
            "z": {"L['z'][0]": (True,)},
        }
        self.assertEqual(result, expected)

    def test_dynamic_tensor_deeply_nested(self):
        # [FUNCTION DOC] Five examples with a tensor nested four dictionaries deep whose
        # size grows. Expects a single entry, with the full nested path as key, flagged
        # dynamic.
        ex1 = {"z": {"z": {"z": {"z": {0: torch.ones(1)}}}}}
        ex2 = {"z": {"z": {"z": {"z": {0: torch.ones(2)}}}}}
        ex3 = {"z": {"z": {"z": {"z": {0: torch.ones(3)}}}}}
        ex4 = {"z": {"z": {"z": {"z": {0: torch.ones(4)}}}}}
        ex5 = {"z": {"z": {"z": {"z": {0: torch.ones(5)}}}}}
        examples = [ex1, ex2, ex3, ex4, ex5]

        result = track_dynamism_across_examples(examples)
        expected = {
            "z": {
                "L['z']['z']['z']['z'][0]": (True,),
            },
        }
        self.assertEqual(result, expected)

    def test_mixed_dynamism(self):
        # [FUNCTION DOC] Five examples mixing a matrix with a growing first dimension, a
        # list holding a growing tensor and a changing integer, and a dictionary holding a
        # constant integer. Expects dynamic flags for the varying parts and false for the
        # constant.
        ex1 = {"a": torch.ones(1, 2), "b": [torch.ones(1), 3], "c": {"d": 42}}
        ex2 = {"a": torch.ones(2, 2), "b": [torch.ones(2), 4], "c": {"d": 42}}
        ex3 = {"a": torch.ones(3, 2), "b": [torch.ones(3), 5], "c": {"d": 42}}
        ex4 = {"a": torch.ones(4, 2), "b": [torch.ones(4), 6], "c": {"d": 42}}
        ex5 = {"a": torch.ones(5, 2), "b": [torch.ones(5), 7], "c": {"d": 42}}
        examples = [ex1, ex2, ex3, ex4, ex5]

        result = track_dynamism_across_examples(examples)
        expected = {
            "a": {"L['a']": (True, False)},
            "b": {"L['b'][0]": (True,), "L['b'][1]": (True,)},
            "c": {"L['c']['d']": (False,)},
        }
        self.assertEqual(result, expected)

    def test_nn_module(self):
        # [FUNCTION DOC] Compares two module instances whose input feature counts differ
        # (3210 versus 3211) but output counts are equal. Expects the weight dimension tied
        # to input features, the in_features and stored scalar attributes to be flagged
        # dynamic and output-related and bias entries static, with paths going through the
        # converted module mapping.
        class Y(torch.nn.Module):
            # [CLASS DOC] HELPER: Inner module holding a linear layer and an integer copy of
            # the input size.
            def __init__(self, n_input, n_output):
                # [FUNCTION DOC] HELPER: Creates a linear layer from input to output sizes
                # and stores the input size in an attribute.
                super().__init__()
                self.compress = torch.nn.Linear(n_input, n_output)
                self.x = n_input

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer and scales by the stored
                # integer.
                return self.compress(x) * self.x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer module wrapping the inner module and recording the
            # input and output sizes as attributes.
            def __init__(self, n_input, n_output):
                # [FUNCTION DOC] HELPER: Stores the sizes as attributes (before calling the
                # base constructor) and creates the inner module.
                self.n_input = n_input
                self.n_output = n_output
                super().__init__()
                self.y = Y(n_input, n_output)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Delegates to the inner module.
                return self.y(x)

        model1 = M(3210, 30)
        model2 = M(3211, 30)

        result = track_dynamism_across_examples(
            [
                {"self": model1},
                {"self": model2},
            ]
        )
        expected = {
            "self": {
                "L['self']['_modules']['y']['_modules']['compress']['_parameters']['weight']": (
                    False,
                    True,
                ),
                "L['self']['_modules']['y']['_modules']['compress']['_parameters']['bias']": (
                    False,
                ),
                "L['self']['_modules']['y']['_modules']['compress']['bias']": (False,),
                "L['self']['_modules']['y']['_modules']['compress']['in_features']": (
                    True,
                ),
                "L['self']['_modules']['y']['_modules']['compress']['out_features']": (
                    False,
                ),
                "L['self']['_modules']['y']['_modules']['compress']['weight']": (
                    False,
                    True,
                ),
                "L['self']['_modules']['y']['x']": (True,),
                "L['self']['n_input']": (True,),
                "L['self']['n_output']": (False,),
            }
        }
        self.assertEqual(result, expected)

    def test_property_not_implemented(self):
        # [FUNCTION DOC] Verifies that a module with a property that raises
        # NotImplementedError does not break tracking: the property is skipped and only the
        # linear layer entries remain, all static since both modules have identical sizes.
        class ModuleWithNotImplementedProperty(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer and a property that always
            # raises NotImplementedError when read.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Creates a linear layer of the given input and
                # output sizes.
                super().__init__()
                self.linear = torch.nn.Linear(x, y)

            @property
            def not_implemented_property(self):
                # [FUNCTION DOC] HELPER: Property that always raises NotImplementedError.
                raise NotImplementedError("This property is not implemented")

        module1 = ModuleWithNotImplementedProperty(10, 10)
        module2 = ModuleWithNotImplementedProperty(10, 10)

        result = track_dynamism_across_examples(
            [
                {"self": module1},
                {"self": module2},
            ]
        )

        expected = {
            "self": {
                "L['self']['_modules']['linear']['_parameters']['weight']": (
                    False,
                    False,
                ),
                "L['self']['_modules']['linear']['_parameters']['bias']": (False,),
                "L['self']['_modules']['linear']['bias']": (False,),
                "L['self']['_modules']['linear']['in_features']": (False,),
                "L['self']['_modules']['linear']['out_features']": (False,),
                "L['self']['_modules']['linear']['weight']": (False, False),
            }
        }

        self.assertEqual(result, expected)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
