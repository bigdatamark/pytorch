# Owner(s): ["module: nn"]
# [MODULE DOC] Tests for torch.nn.init. Defines a table of every public in-place initializer with
# a sample shape and keyword arguments, then tests gain computation, argument validation errors,
# statistical distributions of the initialized values (using scipy Kolmogorov-Smirnov tests when
# scipy is available), special structure (identity, Dirac, orthogonal, sparse), generator support,
# meta-device and zero-element behavior, and the deprecation warning of non-underscore names. The
# device-generic class is instantiated per device at import time; scipy is imported only if
# available.
import math
import random
import string
import unittest
from functools import reduce
from operator import mul

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize as parametrize_test,
    run_tests,
    skipIfNoLapack,
    slowTest,
    TEST_SCIPY,
    TestCase,
)


if TEST_SCIPY:
    from scipy import stats


# (init_fn, shape, extra kwargs) covering every public init function.
# Shapes satisfy each function's dimensionality requirements.
# The 0-element variant is derived by replacing dim-0 with 0.
ALL_INIT_FNS = [
    (init.uniform_, (3, 5), {}),
    (init.normal_, (3, 5), {}),
    (init.trunc_normal_, (3, 5), {}),
    (init.constant_, (3, 5), {"val": 0.3}),
    (init.ones_, (3, 5), {}),
    (init.zeros_, (3, 5), {}),
    (init.eye_, (3, 5), {}),
    (init.dirac_, (3, 16, 5), {}),
    (init.xavier_uniform_, (3, 5), {}),
    (init.xavier_normal_, (3, 5), {}),
    (init.kaiming_uniform_, (3, 5), {}),
    (init.kaiming_normal_, (3, 5), {}),
    (init.orthogonal_, (3, 5), {}),
    (init.sparse_, (3, 5), {"sparsity": 0.1}),
]


class TestNNInit(TestCase):
    # [CLASS DOC] Device-independent tests of torch.nn.init functions on CPU tensors. The setUp
    # hook seeds the Python random module so randomly drawn sizes and parameters are reproducible.
    # Helper methods build random-shaped zero tensors and run scipy goodness-of-fit checks
    # (normal, truncated normal, uniform).
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp and seeds the Python random module with a
        # fixed seed, so random sizes and parameters used by the tests are deterministic.
        super().setUp()
        random.seed(123)

    @torch._dynamo.disable
    def _is_normal(self, tensor, mean, std):
        # [FUNCTION DOC] HELPER: Flattens the tensor to a list and runs a Kolmogorov-Smirnov test
        # against a normal distribution with the given mean and standard deviation. Returns True
        # when the p-value exceeds 0.0001. Needs scipy; excluded from dynamo tracing.
        samples = tensor.view(-1).tolist()
        p_value = stats.kstest(samples, "norm", args=(mean, std))[1]
        return p_value > 0.0001

    @torch._dynamo.disable
    def _is_trunc_normal(self, tensor, mean, std, a, b):
        # [FUNCTION DOC] HELPER: Standardizes the tensor values with the given mean and std,
        # converts the truncation bounds the same way, and runs a Kolmogorov-Smirnov test against
        # a standard truncated normal. Returns True when the p-value exceeds 0.0001. Needs scipy;
        # excluded from dynamo tracing.
        # scipy's trunc norm is suited for data drawn from N(0, 1),
        # so we need to transform our data to test it using scipy.
        z_samples = (tensor.view(-1) - mean) / std
        z_samples = z_samples.tolist()
        a0 = (a - mean) / std
        b0 = (b - mean) / std
        p_value = stats.kstest(z_samples, "truncnorm", args=(a0, b0))[1]
        return p_value > 0.0001

    @torch._dynamo.disable
    def _is_uniform(self, tensor, a, b):
        # [FUNCTION DOC] HELPER: Flattens the tensor to a list and runs a Kolmogorov-Smirnov test
        # against a uniform distribution on the interval from a to b. Returns True when the
        # p-value exceeds 0.0001. Needs scipy; excluded from dynamo tracing.
        samples = tensor.view(-1).tolist()
        p_value = stats.kstest(samples, "uniform", args=(a, (b - a)))[1]
        return p_value > 0.0001

    def _create_random_nd_tensor(self, dims, size_min, size_max):
        # [FUNCTION DOC] HELPER: Returns a float zero tensor with the given number of dimensions,
        # each size drawn at random from the inclusive range between the minimum and maximum.
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = torch.zeros(size)
        return tensor

    def _random_float(self, a, b):
        # [FUNCTION DOC] HELPER: Returns a random float drawn uniformly between a and b using the
        # Python random module.
        return (b - a) * random.random() + a

    def test_calculate_gain_linear(self):
        # [FUNCTION DOC] Asserts calculate_gain returns exactly 1 for linear and convolution names
        # (the conv_transpose2d name is listed twice and conv_transpose1d is not covered).
        for fn in [
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose2d",
            "conv_transpose2d",
            "conv_transpose3d",
        ]:
            gain = init.calculate_gain(fn)
            self.assertEqual(gain, 1)

    def test_calculate_gain_nonlinear(self):
        # [FUNCTION DOC] Asserts calculate_gain for sigmoid, tanh, relu and leaky_relu equals the
        # hard-coded constants (1, 5/3, sqrt 2 and the leaky value for the default slope).
        # KNOWN ISSUE: the final selu branch can never run because selu is not in the tested list.
        for fn in ["sigmoid", "tanh", "relu", "leaky_relu"]:
            gain = init.calculate_gain(fn)
            if fn == "sigmoid":
                self.assertEqual(gain, 1)
            elif fn == "tanh":  # 5 / 3
                self.assertEqual(gain, 1.6666666666666667)
            elif fn == "relu":  # sqrt(2)
                self.assertEqual(gain, 1.4142135623730951)
            elif fn == "leaky_relu":  # sqrt(2 / 1 + slope^2))
                self.assertEqual(gain, 1.4141428569978354)
            elif fn == "selu":
                self.assertEqual(gain, 0.75)

    def test_calculate_gain_leaky_relu(self):
        # [FUNCTION DOC] Asserts calculate_gain for leaky_relu with parameter None, 0, 0.01 and 10
        # returns the expected constants (default slope 0.01; slope 0 equals the relu gain).
        for param in [None, 0, 0.01, 10]:
            gain = init.calculate_gain("leaky_relu", param)
            if param is None:  # Default slope is 0.01
                self.assertEqual(gain, 1.4141428569978354)
            elif param == 0:  # No slope = same gain as normal ReLU
                self.assertEqual(gain, 1.4142135623730951)
            elif param == 0.01:
                self.assertEqual(gain, 1.4141428569978354)
            elif param == 10:
                self.assertEqual(gain, 0.14071950894605836)

    def test_calculate_gain_leaky_relu_only_accepts_numbers(self):
        # [FUNCTION DOC] Asserts calculate_gain for leaky_relu raises ValueError when the
        # parameter is a bool, a list or a dict.
        for param in [True, [1], {"a": "b"}]:
            with self.assertRaises(ValueError):
                init.calculate_gain("leaky_relu", param)

    def test_calculate_gain_only_accepts_valid_nonlinearities(self):
        # [FUNCTION DOC] Asserts calculate_gain raises ValueError for random lowercase strings of
        # lengths 2, 5 and 25, which are not valid nonlinearity names.
        for n in [2, 5, 25]:
            # Generate random strings of lengths that definitely aren't supported
            random_string = "".join(
                [random.choice(string.ascii_lowercase) for i in range(n)]
            )
            with self.assertRaises(ValueError):
                init.calculate_gain(random_string)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_uniform(self):
        # [FUNCTION DOC] Initializes random tensors of 1, 2 and 4 dimensions with uniform_ over a
        # random interval and asserts the values pass the uniform goodness-of-fit check. Skipped
        # without scipy.
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            a = self._random_float(-3, 3)
            b = a + self._random_float(1, 5)
            init.uniform_(input_tensor, a=a, b=b)
            if not self._is_uniform(input_tensor, a, b):
                raise AssertionError("Expected uniform distribution")

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_normal(self):
        # [FUNCTION DOC] Initializes random tensors of 1, 2 and 4 dimensions with normal_ using a
        # random mean and std and asserts the values pass the normal goodness-of-fit check.
        # Skipped without scipy.
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50)
            mean = self._random_float(-3, 3)
            std = self._random_float(1, 5)
            init.normal_(input_tensor, mean=mean, std=std)

            if not self._is_normal(input_tensor, mean, std):
                raise AssertionError("Expected normal distribution")

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_trunc_normal_generator(self):
        # [FUNCTION DOC] Asserts trunc_normal_ driven by an explicit generator seeded with 42
        # gives the same values as the global generator seeded with 42, and that the values pass
        # the truncated-normal check. Skipped without scipy.
        gen = torch.Generator()
        gen.manual_seed(42)
        input_tensor = torch.empty(5)
        init.trunc_normal_(input_tensor, generator=gen)

        ref = torch.empty(5)
        torch.manual_seed(42)
        init.trunc_normal_(ref)

        self.assertEqual(input_tensor, ref)
        if not self._is_trunc_normal(input_tensor, mean=0, std=1, a=0, b=1):
            raise AssertionError("Expected truncated normal distribution")

    def test_init_methods_meta(self):
        # [FUNCTION DOC] For every entry of the initializer table, applies the function to an
        # empty meta tensor of the sample shape inside a subtest and asserts it returns the same
        # tensor object.
        for fn, shape, kwargs in ALL_INIT_FNS:
            with self.subTest(init_fn=fn.__name__):
                t = torch.empty(shape, device="meta")
                result = fn(t, **kwargs)
                self.assertIs(result, t)

    def test_init_methods_meta_zero_element(self):
        # [FUNCTION DOC] Same as the meta test but with the first dimension replaced by zero, so
        # the meta tensors have no elements; asserts each initializer returns its input object.
        for fn, shape, kwargs in ALL_INIT_FNS:
            with self.subTest(init_fn=fn.__name__):
                t = torch.empty((0,) + shape[1:], device="meta")
                result = fn(t, **kwargs)
                self.assertIs(result, t)

    def test_constant(self):
        # [FUNCTION DOC] Asserts constant_ with a random value on tensors of 1, 2 and 4 dimensions
        # equals a tensor filled with that value.
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5)
            val = self._random_float(1, 10)
            init.constant_(input_tensor, val)

            self.assertEqual(input_tensor, input_tensor.clone().fill_(val))

    def test_ones_and_zeros(self):
        # [FUNCTION DOC] Asserts ones_ and zeros_ fill random-shaped tensors of 1, 2 and 4
        # dimensions with 1 and 0 respectively.
        for init_fn_, val in zip([init.ones_, init.zeros_], [1, 0]):
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(
                    dims, size_min=1, size_max=5
                )
                init_fn_(input_tensor)

                self.assertEqual(input_tensor, input_tensor.clone().fill_(val))

    def test_eye(self):
        # [FUNCTION DOC] Applies eye_ to a random 2-D tensor and checks every element: ones on the
        # diagonal and zeros elsewhere; raises AssertionError on a mismatch.
        input_tensor = self._create_random_nd_tensor(2, size_min=1, size_max=5)
        init.eye_(input_tensor)

        # Check every single element
        for i in range(input_tensor.size(0)):
            for j in range(input_tensor.size(1)):
                if i == j:
                    if input_tensor[i][j] != 1:
                        raise AssertionError(
                            f"Expected 1 at [{i}][{j}], got {input_tensor[i][j]}"
                        )
                else:
                    if input_tensor[i][j] != 0:
                        raise AssertionError(
                            f"Expected 0 at [{i}][{j}], got {input_tensor[i][j]}"
                        )

    def test_eye_only_works_on_2d_inputs(self):
        # [FUNCTION DOC] Asserts eye_ raises ValueError for tensors with 1 and 3 dimensions.
        for dims in [1, 3]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.eye_(tensor)

    def test_dirac_properties(self):
        # [FUNCTION DOC] For 3, 4 and 5 dimensional tensors and group counts 1, 2 and 3, applies
        # dirac_ and asserts the number of nonzero entries and the sum of values both equal the
        # smaller of per-group output and input channels times the number of groups.
        for dims in [3, 4, 5]:
            for groups in [1, 2, 3]:
                # prepare random tensor with random sizes, but fits groups
                a, c, d, e = (random.randint(1, 5) for _ in range(4))
                b = random.randint(
                    1, 5 * groups
                )  # same range as a*groups but all range allowed
                # make sure first dim divides by groups
                input_tensor = torch.randn((a * groups, b, c, d, e)[:dims])

                init.dirac_(input_tensor, groups)

                c_out, c_in = input_tensor.size(0) // groups, input_tensor.size(1)
                min_d = min(c_out, c_in)
                # Check number of nonzeros is equivalent to smallest dim (for each group)
                if torch.nonzero(input_tensor).size(0) != min_d * groups:
                    raise AssertionError(
                        f"Expected {min_d * groups} nonzeros, got {torch.nonzero(input_tensor).size(0)}"
                    )
                # Check sum of values (can have precision issues, hence assertEqual) is also equivalent
                self.assertEqual(input_tensor.sum(), min_d * groups)

    def test_dirac_identity(self):
        # [FUNCTION DOC] For 1, 3 groups, builds a Dirac filter for 1D, 2D and 3D convolutions and
        # applies the matching functional convolution to random input. Asserts that each group's
        # input channels appear unchanged in the interior of the output (kernel borders cropped)
        # and that the extra output channels of each group are all zero.
        for groups in [1, 3]:
            batch, in_c, out_c, size, kernel_size = (
                8,
                3,
                9,
                5,
                3,
            )  # in_c, out_c must divide by groups
            eff_out_c = out_c // groups

            # Test 1D
            input_var = torch.randn(batch, in_c, size)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            output_var = F.conv1d(input_var, filter_var)
            input_tensor, output_tensor = (
                input_var.data,
                output_var.data,
            )  # Variables do not support nonzero
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                self.assertEqual(
                    input_tensor[:, :, 1:-1],
                    output_tensor[:, eff_out_c * g : eff_out_c * g + in_c, :],
                )
                # Assert extra outputs are 0
                if (
                    torch.nonzero(
                        output_tensor[:, eff_out_c * g + in_c : eff_out_c * (g + 1), :]
                    ).numel()
                    != 0
                ):
                    raise AssertionError("Expected extra outputs to be 0")

            # Test 2D
            input_var = torch.randn(batch, in_c, size, size)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size, kernel_size)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            output_var = F.conv2d(input_var, filter_var)
            input_tensor, output_tensor = (
                input_var.data,
                output_var.data,
            )  # Variables do not support nonzero
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                self.assertEqual(
                    input_tensor[:, :, 1:-1, 1:-1],
                    output_tensor[:, eff_out_c * g : eff_out_c * g + in_c, :, :],
                )
                # Assert extra outputs are 0
                if (
                    torch.nonzero(
                        output_tensor[
                            :, eff_out_c * g + in_c : eff_out_c * (g + 1), :, :
                        ]
                    ).numel()
                    != 0
                ):
                    raise AssertionError("Expected extra outputs to be 0")

            # Test 3D
            input_var = torch.randn(batch, in_c, size, size, size)
            filter_var = torch.zeros(
                eff_out_c, in_c, kernel_size, kernel_size, kernel_size
            )
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            output_var = F.conv3d(input_var, filter_var)
            input_tensor, output_tensor = input_var.data, output_var.data
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                self.assertEqual(
                    input_tensor[:, :, 1:-1, 1:-1, 1:-1],
                    output_tensor[:, eff_out_c * g : eff_out_c * g + in_c, :, :, :],
                )
                # Assert extra outputs are 0
                if (
                    torch.nonzero(
                        output_tensor[
                            :, eff_out_c * g + in_c : eff_out_c * (g + 1), :, :, :
                        ]
                    ).numel()
                    != 0
                ):
                    raise AssertionError("Expected extra outputs to be 0")

    def test_dirac_only_works_on_3_4_5d_inputs(self):
        # [FUNCTION DOC] Asserts dirac_ raises ValueError for tensors with 1, 2 and 6 dimensions.
        for dims in [1, 2, 6]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.dirac_(tensor)

    def test_xavier_uniform_errors_on_inputs_smaller_than_2d(self):
        # [FUNCTION DOC] Asserts xavier_uniform_ raises ValueError for 0-D and 1-D tensors.
        for dims in [0, 1]:
            tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
            with self.assertRaises(ValueError):
                init.xavier_uniform_(tensor)

    def test_xavier_normal_errors_on_inputs_smaller_than_2d(self):
        # [FUNCTION DOC] Asserts xavier_normal_ raises ValueError for 0-D and 1-D tensors.
        for dims in [0, 1]:
            tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
            with self.assertRaises(ValueError):
                init.xavier_normal_(tensor)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @slowTest
    def test_xavier_uniform(self):
        # [FUNCTION DOC] Applies xavier_uniform_ with and without a random gain on 2-D and 4-D
        # tensors, computes the expected bound from the gain and the fan sizes (including the
        # kernel size for 4-D), and asserts the values are uniform on that symmetric interval.
        # Slow test; skipped without scipy.
        for use_gain in [True, False]:
            for dims in [2, 4]:
                input_tensor = self._create_random_nd_tensor(
                    dims, size_min=20, size_max=25
                )
                gain = 1

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.xavier_uniform_(input_tensor, gain=gain)
                else:
                    init.xavier_uniform_(input_tensor)

                fan_in = input_tensor.size(1)
                fan_out = input_tensor.size(0)
                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                bounds = expected_std * math.sqrt(3)
                if not self._is_uniform(input_tensor, -bounds, bounds):
                    raise AssertionError("Expected uniform distribution")

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_xavier_normal(self):
        # [FUNCTION DOC] Applies xavier_normal_ with and without a random gain on 2-D and 4-D
        # tensors and asserts the values are normal with the expected zero mean and gain-and-fan-
        # based standard deviation. Skipped without scipy.
        for use_gain in [True, False]:
            for dims in [2, 4]:
                input_tensor = self._create_random_nd_tensor(
                    dims, size_min=20, size_max=25
                )
                gain = 1

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.xavier_normal_(input_tensor, gain=gain)
                else:
                    init.xavier_normal_(input_tensor)

                fan_in = input_tensor.size(1)
                fan_out = input_tensor.size(0)
                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                if not self._is_normal(input_tensor, 0, expected_std):
                    raise AssertionError("Expected normal distribution")

    def test_kaiming_uniform_errors_on_inputs_smaller_than_2d(self):
        # [FUNCTION DOC] Asserts kaiming_uniform_ raises ValueError for 0-D and 1-D tensors.
        for dims in [0, 1]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
                init.kaiming_uniform_(tensor)

    def test_kaiming_normal_errors_on_inputs_smaller_than_2d(self):
        # [FUNCTION DOC] Asserts kaiming_normal_ raises ValueError for 0-D and 1-D tensors.
        for dims in [0, 1]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1)
                init.kaiming_normal_(tensor)

    def test_kaiming_uniform_warning_on_0element_tensor(self):
        # [FUNCTION DOC] Asserts kaiming_uniform_ on a zero-element tensor emits a UserWarning
        # that initializing zero-element tensors is a no-op.
        tensor = torch.empty(0, 1)
        with self.assertWarnsRegex(
            UserWarning, "Initializing zero-element tensors is a no-op"
        ):
            _ = init.kaiming_uniform_(tensor)

    def test_kaiming_normal_warning_on_0element_tensor(self):
        # [FUNCTION DOC] Asserts kaiming_normal_ on a zero-element tensor emits a UserWarning that
        # initializing zero-element tensors is a no-op.
        tensor = torch.empty(0, 1)
        with self.assertWarnsRegex(
            UserWarning, "Initializing zero-element tensors is a no-op"
        ):
            _ = init.kaiming_normal_(tensor)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_kaiming_uniform(self):
        # [FUNCTION DOC] Applies kaiming_uniform_ with and without a negative-slope argument, in
        # fan_in and fan_out modes, on 2-D and 4-D tensors, and asserts the values are uniform on
        # the symmetric interval derived from the expected std. Skipped without scipy.
        for use_a in [True, False]:
            for dims in [2, 4]:
                for mode in ["fan_in", "fan_out"]:
                    input_tensor = self._create_random_nd_tensor(
                        dims, size_min=20, size_max=25
                    )
                    if use_a:
                        a = self._random_float(0.1, 2)
                        init.kaiming_uniform_(input_tensor, a=a, mode=mode)
                    else:
                        a = 0
                        init.kaiming_uniform_(input_tensor, mode=mode)

                    fan_in = input_tensor.size(1)
                    fan_out = input_tensor.size(0)
                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    if mode == "fan_in":
                        n = fan_in
                    else:
                        n = fan_out

                    expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                    bounds = expected_std * math.sqrt(3.0)
                    if not self._is_uniform(input_tensor, -bounds, bounds):
                        raise AssertionError("Expected uniform distribution")

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_kaiming_normal(self):
        # [FUNCTION DOC] Applies kaiming_normal_ with and without a negative-slope argument, in
        # fan_in and fan_out modes, on 2-D and 4-D tensors, and asserts the values are normal with
        # the expected std. Skipped without scipy.
        for use_a in [True, False]:
            for dims in [2, 4]:
                for mode in ["fan_in", "fan_out"]:
                    input_tensor = self._create_random_nd_tensor(
                        dims, size_min=20, size_max=25
                    )
                    if use_a:
                        a = self._random_float(0.1, 2)
                        init.kaiming_normal_(input_tensor, a=a, mode=mode)
                    else:
                        a = 0
                        init.kaiming_normal_(input_tensor, mode=mode)

                    fan_in = input_tensor.size(1)
                    fan_out = input_tensor.size(0)
                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    if mode == "fan_in":
                        n = fan_in
                    else:
                        n = fan_out

                    expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                    if not self._is_normal(input_tensor, 0, expected_std):
                        raise AssertionError("Expected normal distribution")

    def test_sparse_only_works_on_2d_inputs(self):
        # [FUNCTION DOC] Asserts sparse_ raises ValueError for tensors with 1 and 3 dimensions.
        for dims in [1, 3]:
            with self.assertRaises(ValueError):
                sparsity = self._random_float(0.1, 0.9)
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3)
                init.sparse_(tensor, sparsity)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    def test_sparse_default_std(self):
        # [FUNCTION DOC] Applies sparse_ with a random sparsity, with the default std or a random
        # std, to a random 2-D tensor. Asserts each column has at least the ceiling of sparsity
        # times rows zeros and the nonzero values are normal with the chosen std. Skipped without
        # scipy.
        for use_random_std in [True, False]:
            input_tensor = self._create_random_nd_tensor(2, size_min=30, size_max=35)
            rows = input_tensor.size(0)
            sparsity = self._random_float(0.1, 0.2)

            std = 0.01  # default std
            if use_random_std:
                std = self._random_float(0.01, 0.2)
                init.sparse_(input_tensor, sparsity=sparsity, std=std)
            else:
                init.sparse_(input_tensor, sparsity=sparsity)

            for col_idx in range(input_tensor.size(1)):
                column = input_tensor[:, col_idx]
                if column[column == 0].nelement() < math.ceil(sparsity * rows):
                    raise AssertionError("Expected more zero elements for sparsity")

            if not self._is_normal(input_tensor[input_tensor != 0], 0, std):
                raise AssertionError(
                    "Expected normal distribution for non-zero elements"
                )

    @skipIfNoLapack
    def test_orthogonal(self):
        # [FUNCTION DOC] Applies orthogonal_ with and without a random gain to tensors of several
        # shapes, flattens each to a matrix, and asserts the Gram matrix of the shorter side
        # equals the identity scaled by the squared gain within a tight tolerance. Skipped without
        # LAPACK.
        for use_gain in [True, False]:
            for tensor_size in [[3, 4], [4, 3], [20, 2, 3, 4], [2, 3, 4, 5]]:
                input_tensor = torch.zeros(tensor_size)
                gain = 1.0

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.orthogonal_(input_tensor, gain=gain)
                else:
                    init.orthogonal_(input_tensor)

                rows, cols = tensor_size[0], reduce(mul, tensor_size[1:])
                flattened_tensor = input_tensor.view(rows, cols)
                if rows > cols:
                    self.assertEqual(
                        torch.mm(flattened_tensor.t(), flattened_tensor),
                        torch.eye(cols) * gain**2,
                        atol=1e-6,
                        rtol=0,
                    )
                else:
                    self.assertEqual(
                        torch.mm(flattened_tensor, flattened_tensor.t()),
                        torch.eye(rows) * gain**2,
                        atol=1e-6,
                        rtol=0,
                    )

    def test_deprecation(self):
        # [FUNCTION DOC] Asserts that calling the old non-underscore normal function emits a
        # FutureWarning mentioning deprecation.
        x = torch.randn(3, 3)

        def fn():
            # [FUNCTION DOC] HELPER: Calls the deprecated normal initializer on the enclosing
            # test's tensor; wrapped so the warning can be asserted.
            init.normal(x)

        with self.assertWarnsRegex(
            FutureWarning,
            "deprecated",
            msg="methods not suffixed with underscore should be deprecated",
        ):
            fn()


class TestNNInitDeviceType(TestCase):
    # [CLASS DOC] Device-generic tests (accelerator hardware class) for truncated-normal
    # initialization across dtypes and for running every initializer on each device; instantiated
    # per device type at import time.
    hw_classification = HardwareClassification.ACCELERATOR

    @torch._dynamo.disable
    def _is_trunc_normal(self, tensor, mean, std, a, b):
        # [FUNCTION DOC] HELPER: Same as the CPU class helper: standardizes samples and bounds,
        # runs a Kolmogorov-Smirnov test against a truncated normal and returns True when the
        # p-value exceeds 0.0001. Needs scipy; excluded from dynamo tracing.
        z_samples = (tensor.view(-1) - mean) / std
        z_samples = z_samples.tolist()
        a0 = (a - mean) / std
        b0 = (b - mean) / std
        p_value = stats.kstest(z_samples, "truncnorm", args=(a0, b0))[1]
        return p_value > 0.0001

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @parametrize_test("dims", [1, 2, 4])
    @parametrize_test(
        "dtype",
        [torch.float32, torch.float64],
    )
    def test_trunc_normal_all_dtypes(self, device, dims, dtype):
        # [FUNCTION DOC] Parametrized over dimension count and float32 or float64. Applies
        # trunc_normal_ with random mean, std and bounds on the device, asserts all values lie
        # within the bounds rounded to the dtype, and that the values pass the truncated-normal
        # check. Skipped without scipy.
        size = [random.randint(30, 50) for _ in range(dims)]
        input_tensor = torch.zeros(size, dtype=dtype, device=device)
        mean = random.uniform(-3, 3)
        std = random.uniform(0.01, 1)
        a = random.uniform(mean - 2 * std, mean)
        b = random.uniform(mean, mean + 2 * std)
        init.trunc_normal_(input_tensor, mean=mean, std=std, a=a, b=b)

        # Compare bounds at tensor precision: a and b get rounded when
        # cast to reduced-precision dtypes.
        lo = torch.tensor(a, dtype=dtype).item()
        hi = torch.tensor(b, dtype=dtype).item()
        self.assertTrue(
            input_tensor.min().item() >= lo,
            lambda msg: f"{msg}\n{dtype}: values below lower bound a={a}",
        )
        self.assertTrue(
            input_tensor.max().item() <= hi,
            lambda msg: f"{msg}\n{dtype}: values above upper bound b={b}",
        )
        self.assertTrue(
            self._is_trunc_normal(input_tensor.float().cpu(), mean, std, a, b),
            lambda msg: f"{msg}\n{dtype}: failed KS test against truncated normal",
        )

    # Reduced-precision KS test uses fixed wide params to avoid random
    # intervals too narrow for the type's representable value count.
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @parametrize_test(
        "dtype",
        [torch.float16, torch.bfloat16],
    )
    def test_trunc_normal_low_precision(self, device, dtype):
        # [FUNCTION DOC] For float16 and bfloat16, applies trunc_normal_ with fixed wide
        # parameters to a 10000-element tensor, asserts the values stay within the bounds and pass
        # the truncated-normal check. Skipped without scipy.
        n = 10000
        mean, std, a, b = 0.0, 1.0, -2.0, 2.0
        t = torch.empty(n, dtype=dtype, device=device)
        init.trunc_normal_(t, mean=mean, std=std, a=a, b=b)

        self.assertTrue(
            t.min().item() >= a,
            lambda msg: f"{msg}\n{dtype}: values below lower bound a={a}",
        )
        self.assertTrue(
            t.max().item() <= b,
            lambda msg: f"{msg}\n{dtype}: values above upper bound b={b}",
        )
        self.assertTrue(
            self._is_trunc_normal(t.float().cpu(), mean, std, a, b),
            lambda msg: f"{msg}\n{dtype}: failed KS test against truncated normal",
        )

    # Test that trunc_normal_ behaves well for narrow interval compared to std.
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found.")
    @parametrize_test(
        "dtype",
        [torch.float32, torch.float64, torch.float16, torch.bfloat16],
    )
    def test_trunc_normal_narrow_interval(self, device, dtype):
        # [FUNCTION DOC] For four float dtypes, applies trunc_normal_ with a narrow interval far
        # from the mean, asserts the values stay within the bounds and pass the truncated-normal
        # check. Skipped without scipy.
        n = 10000
        mean, std, a, b = -3.0, 1.0, -0.5, 0.5
        t = torch.empty(n, dtype=dtype, device=device)
        init.trunc_normal_(t, mean=mean, std=std, a=a, b=b)

        self.assertTrue(
            t.min().item() >= a,
            lambda msg: f"{msg}\n{dtype}: values below lower bound a={a}",
        )
        self.assertTrue(
            t.max().item() <= b,
            lambda msg: f"{msg}\n{dtype}: values above upper bound b={b}",
        )
        self.assertTrue(
            self._is_trunc_normal(t.float().cpu(), mean, std, a, b),
            lambda msg: f"{msg}\n{dtype}: failed KS test against truncated normal",
        )

    # Sanity check for trunc normal to ensure that we sample a decent
    # subset of the value space.
    @parametrize_test(
        "dtype,min_unique",
        [
            (torch.float32, 5000),
            (torch.float64, 5000),
            (torch.float16, 3000),
            (torch.bfloat16, 1000),
        ],
    )
    def test_trunc_normal_unique_values(self, device, dtype, min_unique):
        # [FUNCTION DOC] Applies trunc_normal_ to 10000 elements per dtype, asserts bounds are
        # respected, the count of distinct values exceeds a per-dtype minimum, and no infinities
        # or NaNs are produced.
        n = 10000
        t = torch.empty(n, dtype=dtype, device=device)
        init.trunc_normal_(t, mean=0.0, std=1.0, a=-2.0, b=2.0)

        self.assertTrue(
            t.min().item() >= -2.0,
            lambda msg: f"{msg}\n{dtype}: values below lower bound",
        )
        self.assertTrue(
            t.max().item() <= 2.0,
            lambda msg: f"{msg}\n{dtype}: values above upper bound",
        )

        unique = t.unique().numel()
        self.assertGreater(
            unique,
            min_unique,
            lambda msg: f"{msg}\n{dtype}: only {unique} unique values, expected > {min_unique}",
        )

        self.assertFalse(
            t.isinf().any().item(),
            lambda msg: f"{msg}\n{dtype}: trunc_normal_ produced inf values",
        )
        self.assertFalse(
            t.isnan().any().item(),
            lambda msg: f"{msg}\n{dtype}: trunc_normal_ produced nan values",
        )

    # Sanity check that we don't round to the boundary by mistake.
    @parametrize_test(
        "dtype",
        [torch.float32, torch.float64, torch.float16, torch.bfloat16],
    )
    def test_trunc_normal_no_boundary_values_small_std(self, device, dtype):
        # [FUNCTION DOC] For four float dtypes with a small std, asserts that no sampled value
        # equals either truncation bound exactly.
        t = torch.empty(10000, dtype=dtype, device=device)
        init.trunc_normal_(t, mean=0.0, std=0.1, a=-2.0, b=2.0)

        at_lower = (t == -2.0).sum().item()
        at_upper = (t == 2.0).sum().item()
        self.assertEqual(
            at_lower,
            0,
            lambda msg: f"{msg}\n{dtype}: {at_lower} values clamped to lower bound a=-2.0",
        )
        self.assertEqual(
            at_upper,
            0,
            lambda msg: f"{msg}\n{dtype}: {at_upper} values clamped to upper bound b=2.0",
        )

    def _run_init_test(self, device, zero_element=False):
        # [FUNCTION DOC] HELPER: For each entry of the initializer table, optionally replaces the
        # first dimension by zero, builds an empty tensor on the device, and inside a subtest
        # asserts the initializer returns the same tensor object.
        for fn, shape, kwargs in ALL_INIT_FNS:
            if zero_element:
                shape = (0,) + shape[1:]
            with self.subTest(init_fn=fn.__name__):
                t = torch.empty(shape, device=device)
                result = fn(t, **kwargs)
                self.assertIs(result, t)

    def test_init_methods_no_error(self, device):
        # [FUNCTION DOC] Smoke test: runs every table initializer on the device and checks the in-
        # place return value.
        # Smoke test: every public init function must run without error on all
        # devices (including meta where the ops are no-ops).
        self._run_init_test(device)

    def test_init_methods_zero_element(self, device):
        # [FUNCTION DOC] Runs every table initializer on zero-element tensors on the device and
        # checks the in-place return value.
        # Every init function should handle 0-element tensors as a no-op.
        self._run_init_test(device, zero_element=True)


instantiate_device_type_tests(TestNNInitDeviceType, globals())


if __name__ == "__main__":
    run_tests()
