# Owner(s): ["module: nn"]
# [MODULE DOC] Tests for lazy nn modules (uninitialized parameters and buffers, LazyLinear,
# lazy convolutions, lazy batch and instance norms). Covers state_dict load interplay,
# pickling, jit, share_memory, optimizers, weight/spectral norm and device materialization.
# Instantiates device-generic tests for TestLazyModulesDevice at import time.
import pickle

import torch
import torch.nn as nn
from torch.nn import Buffer, Parameter
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    suppress_warnings,
    TestCase,
)


class LazyModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
    # [CLASS DOC] HELPER: Minimal module combining the lazy mixin with the plain Module base;
    # has no forward and is used to hold manually registered uninitialized parameters and
    # buffers.
    pass


class TestLazyModules(TestCase):
    # [CLASS DOC] Generic-hardware test case for lazy module behavior: state_dict loading,
    # pickling, hooks, per-dimension conv/norm lazy variants, optimizers and invalid operations
    # on uninitialized parameters. Uses the module level LazyModule helper and the _check_*
    # helper methods.
    hw_classification = HardwareClassification.GENERIC

    @suppress_warnings
    def test_lazy_module_parameter(self):
        # [FUNCTION DOC] Checks that an uninitialized parameter shows up in the state dict as
        # an uninitialized parameter, that loading it into a module holding a real parameter
        # raises a RuntimeError about the shape of an uninitialized parameter, that loading a
        # real parameter into the lazy module materializes it with the right values, and that
        # loading uninitialized into uninitialized leaves it uninitialized.
        module = LazyModule()
        module.register_parameter("test_param", UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())
        state_dict = module.state_dict()
        self.assertIsInstance(state_dict["test_param"], UninitializedParameter)
        new_module = LazyModule()
        # An error is raised when there is an attempt to replace an existing parameter
        # with an uninitialized one
        new_module.register_parameter("test_param", nn.Parameter(torch.ones(5, 5)))
        with self.assertRaisesRegex(RuntimeError, "shape of an uninitialized"):
            new_module.load_state_dict(state_dict)
        # Uninitialized parameters are overridden when the state dict to be loaded contains a valid one
        new_module = LazyModule()
        new_module.register_parameter("test_param", nn.Parameter(torch.ones(5, 5)))
        module.load_state_dict(new_module.state_dict())
        self.assertEqual(module.test_param, torch.ones((5, 5)))

        # Uninitialized parameters are left unchanged
        module = LazyModule()
        module.register_parameter("test_param", UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())

        new_module = LazyModule()
        new_module.register_parameter("test_param", UninitializedParameter())
        module.load_state_dict(new_module.state_dict())
        self.assertTrue(module.has_uninitialized_params())

    @suppress_warnings
    def test_lazy_module_buffer(self):
        # [FUNCTION DOC] Buffer analog of the lazy parameter test: uninitialized buffer appears
        # in the state dict, loading it over an initialized buffer raises RuntimeError, loading
        # an initialized buffer into the lazy module fills it, and loading uninitialized over
        # uninitialized (done twice) keeps it uninitialized.
        module = LazyModule()
        module.test_buffer = UninitializedBuffer()
        self.assertTrue(module.has_uninitialized_params())
        state_dict = module.state_dict()
        self.assertIsInstance(state_dict["test_buffer"], UninitializedBuffer)
        new_module = LazyModule()
        # An error is raised when there is an attempt to replace an existing parameter
        # with an uninitialized one
        new_module.test_buffer = Buffer(torch.ones(5, 5))
        with self.assertRaisesRegex(RuntimeError, "shape of an uninitialized"):
            new_module.load_state_dict(state_dict)
        # Uninitialized parameters are overridden when the state dict to be loaded contains a valid one
        new_module = LazyModule()
        new_module.test_buffer = Buffer(torch.ones(5, 5))
        module.load_state_dict(new_module.state_dict())
        self.assertEqual(module.test_buffer, torch.ones((5, 5)))

        # Uninitialized parameters are left unchanged
        module = LazyModule()
        module.test_buffer = UninitializedBuffer()
        self.assertTrue(module.has_uninitialized_params())

        new_module = LazyModule()
        new_module.test_buffer = UninitializedBuffer()
        module.load_state_dict(new_module.state_dict())
        module.load_state_dict(new_module.state_dict())
        self.assertTrue(module.has_uninitialized_params())

    @suppress_warnings
    def test_lazy_module_jit_param(self):
        # [FUNCTION DOC] Asserts that scripting a module with an uninitialized parameter raises
        # a RuntimeError telling the user to run a forward pass first.
        module = LazyModule()
        module.register_parameter("test_param", UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())
        with self.assertRaisesRegex(RuntimeError, "run a forward pass"):
            torch.jit.script(module)

    @suppress_warnings
    def test_lazy_module_jit_buffer(self):
        # [FUNCTION DOC] Asserts that scripting a module with an uninitialized buffer raises a
        # RuntimeError telling the user to run a forward pass first.
        module = LazyModule()
        module.test_buffer = UninitializedBuffer()
        self.assertTrue(module.has_uninitialized_params())
        with self.assertRaisesRegex(RuntimeError, "run a forward pass"):
            torch.jit.script(module)

    @suppress_warnings
    def test_lazy_share_memory_param(self):
        # [FUNCTION DOC] Asserts that share_memory on a module with an uninitialized parameter
        # raises a RuntimeError mentioning an uninitialized tensor.
        module = LazyModule()
        module.register_parameter("test_param", UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())
        with self.assertRaisesRegex(RuntimeError, "share memory on an uninitialized"):
            module.share_memory()

    @suppress_warnings
    def test_lazy_share_memory_buffer(self):
        # [FUNCTION DOC] Asserts that share_memory on a module with an uninitialized buffer
        # raises a RuntimeError mentioning an uninitialized tensor.
        module = LazyModule()
        module.test_buffer = UninitializedBuffer()
        self.assertTrue(module.has_uninitialized_params())
        with self.assertRaisesRegex(RuntimeError, "share memory on an uninitialized"):
            module.share_memory()

    @suppress_warnings
    def test_linear(self):
        # [FUNCTION DOC] Runs a LazyLinear once on a 5 by 5 ones input and checks it turned
        # into a plain Linear (no longer a LazyLinear) with weight shape 10 by 5 and bias shape
        # 10, nonzero initialized values and nonzero output, and that a second forward matches
        # the functional linear op.
        module = nn.LazyLinear(10)
        self.assertIsInstance(module.weight, UninitializedParameter)
        self.assertIsInstance(module.bias, UninitializedParameter)
        input = torch.ones(5, 5)
        output = module(input)
        self.assertIsInstance(module, nn.Linear)
        self.assertNotIsInstance(module, nn.LazyLinear)
        self.assertTrue(module.weight.shape == (10, 5))
        self.assertTrue(module.bias.shape == (10,))
        self.assertTrue((module.weight != 0).any())
        self.assertTrue((module.bias != 0).any())
        self.assertTrue((output != 0).any())
        y = module(input)
        self.assertTrue(
            torch.equal(
                torch.nn.functional.linear(input, module.weight, module.bias), y
            )
        )

    @suppress_warnings
    def test_lazy_linear_pickle(self):
        # [FUNCTION DOC] Pickles a fresh LazyLinear and checks it stays a LazyLinear with
        # uninitialized parameters; after a forward pass it is pickled again and the result
        # must be a plain Linear with materialized weight and bias of the expected shapes.
        module = nn.LazyLinear(10)
        self.assertIsInstance(module.weight, UninitializedParameter)
        self.assertIsInstance(module.bias, UninitializedParameter)
        module = pickle.loads(pickle.dumps(module))
        self.assertIsInstance(module, nn.LazyLinear)
        self.assertIsInstance(module.weight, UninitializedParameter)
        self.assertIsInstance(module.bias, UninitializedParameter)
        input = torch.ones(5, 5)
        module(input)  # fully materialized
        new_module = pickle.loads(pickle.dumps(module))
        self.assertIsInstance(new_module, nn.Linear)
        self.assertNotIsInstance(new_module, nn.LazyLinear)
        self.assertTrue(new_module.weight.shape == (10, 5))
        self.assertNotIsInstance(new_module.weight, UninitializedParameter)
        self.assertTrue(new_module.bias.shape == (10,))
        self.assertNotIsInstance(new_module.bias, UninitializedParameter)

    @suppress_warnings
    def test_linear_state(self):
        # [FUNCTION DOC] Loads a regular Linear state dict into a LazyLinear and checks the
        # parameters get initialized with correct shapes; the reverse direction (loading a lazy
        # state dict into a regular Linear) must raise a RuntimeError about an uninitialized
        # shape.
        module = nn.Linear(5, 10)
        lazy_module = nn.LazyLinear(10)
        lazy_module.load_state_dict(module.state_dict())
        # Parameters have been initialized but the module won't become a full
        # Linear one until the first iteration. This is due to
        # limitations on the state_dict loading logic
        self.assertFalse(lazy_module.has_uninitialized_params())
        self.assertTrue(lazy_module.weight.shape == (10, 5))
        self.assertTrue(lazy_module.bias.shape == (10,))

        module = nn.Linear(5, 10)
        lazy_module = nn.LazyLinear(10)
        with self.assertRaisesRegex(RuntimeError, "shape of an uninitialized"):
            module.load_state_dict(lazy_module.state_dict())

    @suppress_warnings
    def test_lazy_linear_state_and_forward(self):
        # [FUNCTION DOC] After loading a state dict, a LazyLinear has no uninitialized
        # parameters but is still a LazyLinear instance; after the first forward it is no
        # longer a LazyLinear and has in_features equal to 5.
        module = nn.Linear(5, 10)
        lazy_module = nn.LazyLinear(10)
        lazy_module.load_state_dict(module.state_dict())
        # Parameters have been initialized but the module won't become a full
        # Linear one until the first iteration. This is due to
        # limitations on the state_dict loading logic
        self.assertFalse(lazy_module.has_uninitialized_params())
        self.assertTrue(isinstance(lazy_module, nn.LazyLinear))

        input = torch.randn(5, 5)
        lazy_module(input)
        self.assertFalse(isinstance(lazy_module, nn.LazyLinear))
        self.assertTrue(lazy_module.in_features == 5)

    def _check_lazy_conv(
        self,
        cls,
        lazy_cls,
        func,
        init_args,
        input_shape,
        expected_weight_shape,
        expected_bias_shape,
        *forward_args,
        **forward_kwargs,
    ):
        # [FUNCTION DOC] HELPER: Shared check for lazy convolutions. Builds the lazy class with
        # the init args, verifies uninitialized weight (and bias if present), runs forward with
        # the given extra args, then verifies the module became the regular class with the
        # expected weight and bias shapes, and that a second forward equals the given
        # functional op applied to the module's weight and bias.
        module = lazy_cls(*init_args)
        self.assertIsInstance(module.weight, UninitializedParameter)
        if module.bias is not None:
            self.assertIsInstance(module.bias, UninitializedParameter)
        input = torch.ones(*input_shape)
        module(input, *forward_args, **forward_kwargs)
        self.assertIsInstance(module, cls)
        self.assertNotIsInstance(module, lazy_cls)
        self.assertEqual(module.weight.shape, expected_weight_shape)
        if module.bias is not None:
            self.assertEqual(module.bias.shape, expected_bias_shape)
        y = module(input)
        self.assertTrue(torch.equal(func(input, module.weight, module.bias), y))

    def _check_lazy_conv_pickle(
        self,
        cls,
        lazy_cls,
        init_args,
        input_shape,
        expected_weight_shape,
        expected_bias_shape,
    ):
        # [FUNCTION DOC] HELPER: Shared pickle check for lazy convolutions. Round trips an
        # unmaterialized lazy module through pickle (stays lazy with uninitialized parameters),
        # runs a forward pass, pickles again, and checks the result is the regular class with
        # materialized parameters of the expected shapes.
        module = lazy_cls(*init_args)
        self.assertIsInstance(module.weight, UninitializedParameter)
        if module.bias is not None:
            self.assertIsInstance(module.bias, UninitializedParameter)
        module = pickle.loads(pickle.dumps(module))
        self.assertIsInstance(module, lazy_cls)
        self.assertIsInstance(module.weight, UninitializedParameter)
        if module.bias is not None:
            self.assertIsInstance(module.bias, UninitializedParameter)
        input = torch.ones(*input_shape)
        module(input)  # fully materialized
        new_module = pickle.loads(pickle.dumps(module))
        self.assertIsInstance(new_module, cls)
        self.assertNotIsInstance(new_module, lazy_cls)
        self.assertEqual(new_module.weight.shape, expected_weight_shape)
        self.assertNotIsInstance(new_module.weight, UninitializedParameter)
        if new_module.bias is not None:
            self.assertEqual(new_module.bias.shape, expected_bias_shape)
            self.assertNotIsInstance(new_module.bias, UninitializedParameter)

    def _check_lazy_conv_state(
        self, gen_module, gen_lazy_module, expected_weight_shape, expected_bias_shape
    ):
        # [FUNCTION DOC] HELPER: Shared state_dict check for lazy convolutions. Loading a
        # regular module's state dict into the lazy module must initialize weight (and bias)
        # with the expected shapes; loading a lazy state dict into a regular module must raise
        # a RuntimeError about an uninitialized shape.
        module = gen_module()
        lazy_module = gen_lazy_module()
        lazy_module.load_state_dict(module.state_dict())
        # Parameters have been initialized but the module won't become a full
        # Conv one until the first iteration. This is due to
        # limitations on the state_dict loading logic
        self.assertFalse(lazy_module.has_uninitialized_params())
        self.assertEqual(lazy_module.weight.shape, expected_weight_shape)
        if lazy_module.bias is not None:
            self.assertEqual(lazy_module.bias.shape, expected_bias_shape)

        module = gen_module()
        lazy_module = gen_lazy_module()
        with self.assertRaisesRegex(RuntimeError, "shape of an uninitialized"):
            module.load_state_dict(lazy_module.state_dict())

    def test_lazy_pre_forward_hook(self):
        # [FUNCTION DOC] Checks that a lazy module still runs a user forward pre-hook: the hook
        # adds one to the input, and the forward returns its input, so zeros come out as ones.
        """
        This test is to test whether lazymodule can register other pre-forward hook
        functions successfully.
        """

        class TestModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            # [CLASS DOC] HELPER: Lazy module whose parameter initialization does nothing and
            # whose forward returns its input unchanged.
            def initialize_parameters(self, input):
                # [FUNCTION DOC] HELPER: No-op initialization of parameters; returns None.
                return None

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return input

        def hook_function(module, input):
            # [FUNCTION DOC] HELPER: Forward pre-hook that returns the first input element plus
            # one, replacing the module input.
            return input[0] + 1

        module = TestModule()
        module.register_forward_pre_hook(hook_function)
        output = module(torch.zeros(2, 2))
        self.assertEqual(output, torch.ones(2, 2))

    def test_lazy_forward_hook(self):
        # [FUNCTION DOC] Checks that a lazy module still runs a user forward hook: the hook
        # returns the first input plus one, which replaces the output, so zeros input yields
        # ones.
        """
        This test is to test whether lazymodule can register other forward hook
        functions successfully.
        """

        class TestModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            # [CLASS DOC] HELPER: Lazy module whose parameter initialization does nothing and
            # whose forward returns its input unchanged.
            def initialize_parameters(self, input):
                # [FUNCTION DOC] HELPER: No-op initialization of parameters; returns None.
                return None

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return input

        def hook_function(module, input, output):
            # [FUNCTION DOC] HELPER: Forward hook that returns the first input element plus
            # one, replacing the module output.
            return input[0] + 1

        module = TestModule()
        module.register_forward_hook(hook_function)
        output = module(torch.zeros(2, 2))
        self.assertEqual(output, torch.ones(2, 2))

    @suppress_warnings
    def test_lazy_conv1d(self):
        # [FUNCTION DOC] Runs the shared lazy convolution check for LazyConv1d with output
        # channels 32, kernel 2, input of shape 192 by 16 by 50, expecting weight shape 32 by
        # 16 by 2.
        self._check_lazy_conv(
            nn.Conv1d,
            nn.LazyConv1d,
            torch.nn.functional.conv1d,
            (32, 2),
            (192, 16, 50),
            (32, 16, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv1d_pickle(self):
        # [FUNCTION DOC] Runs the shared lazy convolution pickle check for LazyConv1d.
        self._check_lazy_conv_pickle(
            nn.Conv1d, nn.LazyConv1d, (32, 2), (192, 16, 50), (32, 16, 2), (32,)
        )

    @suppress_warnings
    def test_lazy_conv1d_state(self):
        # [FUNCTION DOC] Runs the shared lazy convolution state_dict check for Conv1d versus
        # LazyConv1d with 16 input and 32 output channels.
        self._check_lazy_conv_state(
            lambda: nn.Conv1d(16, 32, 2),
            lambda: nn.LazyConv1d(32, 2),
            (32, 16, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv2d(self):
        # [FUNCTION DOC] Runs the shared lazy convolution check for LazyConv2d with input of
        # shape 192 by 16 by 8 by 6, expecting weight shape 32 by 16 by 2 by 2.
        self._check_lazy_conv(
            nn.Conv2d,
            nn.LazyConv2d,
            torch.nn.functional.conv2d,
            (32, 2),
            (192, 16, 8, 6),
            (32, 16, 2, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv2d_pickle(self):
        # [FUNCTION DOC] Runs the shared lazy convolution pickle check for LazyConv2d.
        self._check_lazy_conv_pickle(
            nn.Conv2d, nn.LazyConv2d, (32, 2), (192, 16, 8, 6), (32, 16, 2, 2), (32,)
        )

    @suppress_warnings
    def test_lazy_conv2d_state(self):
        # [FUNCTION DOC] Runs the shared lazy convolution state_dict check for Conv2d versus
        # LazyConv2d.
        self._check_lazy_conv_state(
            lambda: nn.Conv2d(16, 32, 2),
            lambda: nn.LazyConv2d(32, 2),
            (32, 16, 2, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv3d(self):
        # [FUNCTION DOC] Runs the shared lazy convolution check for LazyConv3d with a five
        # dimensional input, expecting weight shape 32 by 16 by 2 by 2 by 2.
        self._check_lazy_conv(
            nn.Conv3d,
            nn.LazyConv3d,
            torch.nn.functional.conv3d,
            (32, 2),
            (192, 16, 8, 7, 6),
            (32, 16, 2, 2, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv3d_pickle(self):
        # [FUNCTION DOC] Runs the shared lazy convolution pickle check for LazyConv3d.
        self._check_lazy_conv_pickle(
            nn.Conv3d,
            nn.LazyConv3d,
            (32, 2),
            (192, 16, 8, 7, 6),
            (32, 16, 2, 2, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv3d_state(self):
        # [FUNCTION DOC] Runs the shared lazy convolution state_dict check for Conv3d versus
        # LazyConv3d.
        self._check_lazy_conv_state(
            lambda: nn.Conv3d(16, 32, 2),
            lambda: nn.LazyConv3d(32, 2),
            (32, 16, 2, 2, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv_transposed1d(self):
        # [FUNCTION DOC] Runs the shared lazy convolution check for LazyConvTranspose1d; the
        # expected weight shape is 16 by 32 by 2 (in channels first).
        self._check_lazy_conv(
            nn.ConvTranspose1d,
            nn.LazyConvTranspose1d,
            torch.nn.functional.conv_transpose1d,
            (32, 2),
            (192, 16, 50),
            (16, 32, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv_transpose1d_kwargs(self):
        # [FUNCTION DOC] Same as the plain transposed 1d check but passes an extra keyword
        # output_size through to the forward call.
        self._check_lazy_conv(
            nn.ConvTranspose1d,
            nn.LazyConvTranspose1d,
            torch.nn.functional.conv_transpose1d,
            (32, 2),
            (192, 16, 50),
            (16, 32, 2),
            (32,),
            output_size=(51,),
        )

    @suppress_warnings
    def test_lazy_conv_transpose1d_pickle(self):
        # [FUNCTION DOC] Runs the shared lazy convolution pickle check for LazyConvTranspose1d.
        self._check_lazy_conv_pickle(
            nn.ConvTranspose1d,
            nn.LazyConvTranspose1d,
            (32, 2),
            (192, 16, 50),
            (16, 32, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv_transpose1d_state(self):
        # [FUNCTION DOC] Runs the shared lazy convolution state_dict check for ConvTranspose1d
        # versus its lazy variant.
        self._check_lazy_conv_state(
            lambda: nn.ConvTranspose1d(16, 32, 2),
            lambda: nn.LazyConvTranspose1d(32, 2),
            (16, 32, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv_transpose2d(self):
        # [FUNCTION DOC] Runs the shared lazy convolution check for LazyConvTranspose2d,
        # expecting weight shape 16 by 32 by 2 by 2.
        self._check_lazy_conv(
            nn.ConvTranspose2d,
            nn.LazyConvTranspose2d,
            torch.nn.functional.conv_transpose2d,
            (32, 2),
            (192, 16, 8, 6),
            (16, 32, 2, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv_transpose2d_kwargs(self):
        # [FUNCTION DOC] Same as the plain transposed 2d check but passes an extra keyword
        # output_size through to the forward call.
        self._check_lazy_conv(
            nn.ConvTranspose2d,
            nn.LazyConvTranspose2d,
            torch.nn.functional.conv_transpose2d,
            (32, 2),
            (192, 16, 8, 6),
            (16, 32, 2, 2),
            (32,),
            output_size=(9, 7),
        )

    @suppress_warnings
    def test_lazy_conv_transpose2d_pickle(self):
        # [FUNCTION DOC] Runs the shared lazy convolution pickle check for LazyConvTranspose2d.
        self._check_lazy_conv_pickle(
            nn.ConvTranspose2d,
            nn.LazyConvTranspose2d,
            (32, 2),
            (192, 16, 8, 6),
            (16, 32, 2, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv_transpose2d_state(self):
        # [FUNCTION DOC] Runs the shared lazy convolution state_dict check for ConvTranspose2d
        # versus its lazy variant.
        self._check_lazy_conv_state(
            lambda: nn.ConvTranspose2d(16, 32, 2),
            lambda: nn.LazyConvTranspose2d(32, 2),
            (16, 32, 2, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv_transpose3d(self):
        # [FUNCTION DOC] Runs the shared lazy convolution check for LazyConvTranspose3d,
        # expecting weight shape 16 by 32 by 2 by 2 by 2.
        self._check_lazy_conv(
            nn.ConvTranspose3d,
            nn.LazyConvTranspose3d,
            torch.nn.functional.conv_transpose3d,
            (32, 2),
            (192, 16, 8, 7, 6),
            (16, 32, 2, 2, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv_transpose3d_kwargs(self):
        # [FUNCTION DOC] Same as the plain transposed 3d check but passes an extra keyword
        # output_size through to the forward call.
        self._check_lazy_conv(
            nn.ConvTranspose3d,
            nn.LazyConvTranspose3d,
            torch.nn.functional.conv_transpose3d,
            (32, 2),
            (192, 16, 8, 7, 6),
            (16, 32, 2, 2, 2),
            (32,),
            output_size=(9, 8, 7),
        )

    @suppress_warnings
    def test_lazy_conv_transpose3d_pickle(self):
        # [FUNCTION DOC] Runs the shared lazy convolution pickle check for LazyConvTranspose3d.
        self._check_lazy_conv_pickle(
            nn.ConvTranspose3d,
            nn.LazyConvTranspose3d,
            (32, 2),
            (192, 16, 8, 7, 6),
            (16, 32, 2, 2, 2),
            (32,),
        )

    @suppress_warnings
    def test_lazy_conv_transpose3d_state(self):
        # [FUNCTION DOC] Runs the shared lazy convolution state_dict check for ConvTranspose3d
        # versus its lazy variant.
        self._check_lazy_conv_state(
            lambda: nn.ConvTranspose3d(16, 32, 2),
            lambda: nn.LazyConvTranspose3d(32, 2),
            (16, 32, 2, 2, 2),
            (32,),
        )

    def _check_lazy_norm(self, cls, lazy_cls, input_shape):
        # [FUNCTION DOC] HELPER: Shared lazy norm check. For every combination of affine and
        # track_running_stats, builds the lazy class, checks the expected uninitialized
        # parameters and buffers, runs forward on a ones input, verifies it became the regular
        # class, then compares output and every weight, bias, running statistic and batch
        # counter against a regular module built with the feature count taken from input
        # dimension one.
        for affine in [False, True]:
            for track_running_stats in [False, True]:
                lazy_module = lazy_cls(
                    affine=affine, track_running_stats=track_running_stats
                )

                if affine:
                    self.assertIsInstance(lazy_module.weight, UninitializedParameter)
                    self.assertIsInstance(lazy_module.bias, UninitializedParameter)
                if track_running_stats:
                    self.assertIsInstance(lazy_module.running_mean, UninitializedBuffer)
                    self.assertIsInstance(lazy_module.running_var, UninitializedBuffer)

                input = torch.ones(*input_shape)
                lazy_output = lazy_module(input)
                self.assertIsInstance(lazy_module, cls)
                self.assertNotIsInstance(lazy_module, lazy_cls)

                num_features = input_shape[1]
                module = cls(
                    num_features, affine=affine, track_running_stats=track_running_stats
                )
                expected_output = module(input)

                self.assertEqual(lazy_output, expected_output)
                if module.weight is not None:
                    self.assertEqual(lazy_module.weight.shape, module.weight.shape)
                    self.assertEqual(lazy_module.weight, module.weight)
                if module.bias is not None:
                    self.assertEqual(lazy_module.bias.shape, module.bias.shape)
                    self.assertEqual(lazy_module.bias, module.bias)
                if module.running_mean is not None:
                    self.assertEqual(
                        lazy_module.running_mean.shape, module.running_mean.shape
                    )
                    self.assertEqual(lazy_module.running_mean, module.running_mean)
                if module.running_var is not None:
                    self.assertEqual(
                        lazy_module.running_var.shape, module.running_var.shape
                    )
                    self.assertEqual(lazy_module.running_var, module.running_var)
                if module.num_batches_tracked is not None:
                    self.assertEqual(
                        lazy_module.num_batches_tracked.shape,
                        module.num_batches_tracked.shape,
                    )
                    self.assertEqual(
                        lazy_module.num_batches_tracked, module.num_batches_tracked
                    )

    def _check_lazy_norm_pickle(self, cls, lazy_cls, input_shape):
        # [FUNCTION DOC] HELPER: Shared lazy norm pickle check. For each affine and
        # track_running_stats combination, a fresh lazy module survives a pickle round trip
        # uninitialized; after a forward pass a second round trip yields the regular class with
        # materialized parameters and buffers.
        for affine in [False, True]:
            for track_running_stats in [False, True]:
                module = lazy_cls(
                    affine=affine, track_running_stats=track_running_stats
                )
                module = pickle.loads(pickle.dumps(module))

                self.assertIsInstance(module, lazy_cls)
                if affine:
                    self.assertIsInstance(module.weight, UninitializedParameter)
                    self.assertIsInstance(module.bias, UninitializedParameter)
                if track_running_stats:
                    self.assertIsInstance(module.running_mean, UninitializedBuffer)
                    self.assertIsInstance(module.running_var, UninitializedBuffer)

                input = torch.ones(*input_shape)
                module(input)  # fully materialized
                module = pickle.loads(pickle.dumps(module))

                self.assertNotIsInstance(module, lazy_cls)
                self.assertIsInstance(module, cls)
                if affine:
                    self.assertNotIsInstance(module.weight, UninitializedParameter)
                    self.assertNotIsInstance(module.bias, UninitializedParameter)
                if track_running_stats:
                    self.assertNotIsInstance(module.running_mean, UninitializedBuffer)
                    self.assertNotIsInstance(module.running_var, UninitializedBuffer)

    def _check_lazy_batchnorm_state(self, cls, lazy_cls):
        # [FUNCTION DOC] HELPER: Shared batch norm state check. Loading a regular state dict
        # with 10 features into an affine, stat-tracking lazy module initializes everything
        # with shape 10 and leaves no uninitialized parameters; loading a lazy state dict into
        # a regular module raises a RuntimeError about an uninitialized shape.
        module = cls(10)
        lazy_module = lazy_cls(affine=True, track_running_stats=True)
        lazy_module.load_state_dict(module.state_dict())
        # Parameters have been initialized but the module won't become a full
        # Conv one until the first iteration. This is due to
        # limitations on the state_dict loading logic
        self.assertFalse(lazy_module.has_uninitialized_params())
        self.assertEqual(lazy_module.weight.shape, (10,))
        self.assertEqual(lazy_module.bias.shape, (10,))
        self.assertEqual(lazy_module.running_mean.shape, (10,))
        self.assertEqual(lazy_module.running_var.shape, (10,))

        module = cls(10)
        lazy_module = lazy_cls()
        with self.assertRaisesRegex(RuntimeError, "shape of an uninitialized"):
            module.load_state_dict(lazy_module.state_dict())

    def _check_lazy_instancenorm_state(self, cls, lazy_cls):
        # [FUNCTION DOC] HELPER: Shared instance norm state check. For each affine and
        # track_running_stats combination, loads a regular state dict into the lazy module and
        # checks shapes of the present parameters and buffers; finally verifies that loading a
        # lazy state dict into a regular module raises a RuntimeError.
        for affine in [False, True]:
            for track_running_stats in [False, True]:
                module = cls(10, affine=affine, track_running_stats=track_running_stats)
                lazy_module = lazy_cls(
                    affine=affine, track_running_stats=track_running_stats
                )
                lazy_module.load_state_dict(module.state_dict())
                # Parameters have been initialized but the module won't become a full
                # InstanceNorm one until the first iteration. This is due to
                # limitations on the state_dict loading logic
                self.assertFalse(lazy_module.has_uninitialized_params())
                if affine:
                    self.assertEqual(lazy_module.weight.shape, (10,))
                    self.assertEqual(lazy_module.bias.shape, (10,))
                if track_running_stats:
                    self.assertEqual(lazy_module.running_mean.shape, (10,))
                    self.assertEqual(lazy_module.running_var.shape, (10,))

        module = cls(10, affine=True, track_running_stats=True)
        lazy_module = lazy_cls(affine=True, track_running_stats=True)
        with self.assertRaisesRegex(RuntimeError, "shape of an uninitialized"):
            module.load_state_dict(lazy_module.state_dict())

    def _check_lazy_norm_with_dict_input(self, cls, lazy_cls, input_shape):
        # [FUNCTION DOC] HELPER: Passes the input as a keyword argument dictionary to a lazy
        # norm and to a regular norm with matching features, and checks equal outputs (lazy
        # forward must support keyword input).
        input = {"input": torch.ones(*input_shape)}

        lazy_module = lazy_cls()
        lazy_output = lazy_module(**input)

        num_features = input_shape[1]
        module = cls(num_features)
        expected_output = module(**input)

        self.assertEqual(lazy_output, expected_output)

    def test_lazy_batchnorm1d(self):
        # [FUNCTION DOC] Runs the lazy norm check for BatchNorm1d on three dimensional and two
        # dimensional inputs.
        self._check_lazy_norm(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 3, 6))
        self._check_lazy_norm(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 6))

    def test_lazy_batchnorm1d_pickle(self):
        # [FUNCTION DOC] Runs the lazy norm pickle check for BatchNorm1d on three and two
        # dimensional inputs.
        self._check_lazy_norm_pickle(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 3, 6))
        self._check_lazy_norm_pickle(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 6))

    def test_lazy_batchnorm1d_state(self):
        # [FUNCTION DOC] Runs the batch norm state check for BatchNorm1d twice (the repeat is
        # redundant).
        self._check_lazy_batchnorm_state(nn.BatchNorm1d, nn.LazyBatchNorm1d)
        self._check_lazy_batchnorm_state(nn.BatchNorm1d, nn.LazyBatchNorm1d)

    def test_lazy_batchnorm2d(self):
        # [FUNCTION DOC] Runs the lazy norm check for BatchNorm2d on a four dimensional input.
        self._check_lazy_norm(nn.BatchNorm2d, nn.LazyBatchNorm2d, (16, 3, 6, 7))

    def test_lazy_batchnorm2d_pickle(self):
        # [FUNCTION DOC] Runs the lazy norm pickle check for BatchNorm2d.
        self._check_lazy_norm_pickle(nn.BatchNorm2d, nn.LazyBatchNorm2d, (16, 3, 6, 7))

    def test_lazy_batchnorm2d_state(self):
        # [FUNCTION DOC] Runs the batch norm state check for BatchNorm2d twice (the repeat is
        # redundant).
        self._check_lazy_batchnorm_state(nn.BatchNorm2d, nn.LazyBatchNorm2d)
        self._check_lazy_batchnorm_state(nn.BatchNorm2d, nn.LazyBatchNorm2d)

    def test_lazy_batchnorm3d(self):
        # [FUNCTION DOC] Runs the lazy norm check for BatchNorm3d on a five dimensional input.
        self._check_lazy_norm(nn.BatchNorm3d, nn.LazyBatchNorm3d, (16, 3, 6, 7, 8))

    def test_lazy_batchnorm3d_pickle(self):
        # [FUNCTION DOC] Runs the lazy norm pickle check for BatchNorm3d.
        self._check_lazy_norm_pickle(
            nn.BatchNorm3d, nn.LazyBatchNorm3d, (16, 3, 6, 7, 8)
        )

    def test_lazy_batchnorm3d_state(self):
        # [FUNCTION DOC] Runs the batch norm state check for BatchNorm3d twice (the repeat is
        # redundant).
        self._check_lazy_batchnorm_state(nn.BatchNorm3d, nn.LazyBatchNorm3d)
        self._check_lazy_batchnorm_state(nn.BatchNorm3d, nn.LazyBatchNorm3d)

    def test_lazy_instancenorm1d(self):
        # [FUNCTION DOC] Runs the lazy norm check for InstanceNorm1d on a three dimensional
        # input.
        self._check_lazy_norm(nn.InstanceNorm1d, nn.LazyInstanceNorm1d, (16, 3, 6))

    def test_lazy_instancenorm1d_pickle(self):
        # [FUNCTION DOC] Runs the lazy norm pickle check for InstanceNorm1d.
        self._check_lazy_norm_pickle(
            nn.InstanceNorm1d, nn.LazyInstanceNorm1d, (16, 3, 6)
        )

    def test_lazy_instancenorm1d_state(self):
        # [FUNCTION DOC] Runs the instance norm state check for InstanceNorm1d twice (the
        # repeat is redundant).
        self._check_lazy_instancenorm_state(nn.InstanceNorm1d, nn.LazyInstanceNorm1d)
        self._check_lazy_instancenorm_state(nn.InstanceNorm1d, nn.LazyInstanceNorm1d)

    def test_lazy_instancenorm2d(self):
        # [FUNCTION DOC] Runs the lazy norm check for InstanceNorm2d on a four dimensional
        # input.
        self._check_lazy_norm(nn.InstanceNorm2d, nn.LazyInstanceNorm2d, (16, 3, 6, 7))

    def test_lazy_instancenorm2d_pickle(self):
        # [FUNCTION DOC] Runs the lazy norm pickle check for InstanceNorm2d.
        self._check_lazy_norm_pickle(
            nn.InstanceNorm2d, nn.LazyInstanceNorm2d, (16, 3, 6, 7)
        )

    def test_lazy_instancenorm2d_state(self):
        # [FUNCTION DOC] Runs the instance norm state check for InstanceNorm2d twice (the
        # repeat is redundant).
        self._check_lazy_instancenorm_state(nn.InstanceNorm2d, nn.LazyInstanceNorm2d)
        self._check_lazy_instancenorm_state(nn.InstanceNorm2d, nn.LazyInstanceNorm2d)

    def test_lazy_instancenorm3d(self):
        # [FUNCTION DOC] Runs the lazy norm check for InstanceNorm3d on a five dimensional
        # input.
        self._check_lazy_norm(
            nn.InstanceNorm3d, nn.LazyInstanceNorm3d, (16, 3, 6, 7, 8)
        )

    def test_lazy_instancenorm3d_pickle(self):
        # [FUNCTION DOC] Runs the lazy norm pickle check for InstanceNorm3d.
        self._check_lazy_norm_pickle(
            nn.InstanceNorm3d, nn.LazyInstanceNorm3d, (16, 3, 6, 7, 8)
        )

    def test_lazy_instancenorm3d_state(self):
        # [FUNCTION DOC] Runs the instance norm state check for InstanceNorm3d twice (the
        # repeat is redundant).
        self._check_lazy_instancenorm_state(nn.InstanceNorm3d, nn.LazyInstanceNorm3d)
        self._check_lazy_instancenorm_state(nn.InstanceNorm3d, nn.LazyInstanceNorm3d)

    def test_lazy_batchnorm_with_dict_input(self):
        # [FUNCTION DOC] Runs the keyword dictionary input check for the lazy BatchNorm
        # variants of dimensions 1, 2 and 3.
        self._check_lazy_norm_with_dict_input(
            nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 3, 6)
        )
        self._check_lazy_norm_with_dict_input(
            nn.BatchNorm2d, nn.LazyBatchNorm2d, (16, 3, 6, 7)
        )
        self._check_lazy_norm_with_dict_input(
            nn.BatchNorm3d, nn.LazyBatchNorm3d, (16, 3, 6, 7, 8)
        )

    @suppress_warnings
    def test_materialize_dtype(self):
        # [FUNCTION DOC] Checks that materializing an uninitialized parameter uses the default
        # dtype, and that after converting the module to half precision materialization
        # produces float16.
        module = LazyModule()
        module.register_parameter("test_param", UninitializedParameter())
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.dtype == torch.get_default_dtype())
        module = LazyModule()
        module.register_parameter("test_param", UninitializedParameter())
        module.half()
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.dtype == torch.float16)

    @suppress_warnings
    def test_chained_initialization(self):
        # [FUNCTION DOC] Builds a network of two stacked LazyLinear layers with output sizes 15
        # and 10, runs one forward on a 5 by 10 input, and checks both layers materialized with
        # weight shapes 15 by 10 and 10 by 15 and matching biases.
        class MyNetwork(torch.nn.Module):
            # [CLASS DOC] HELPER: Small two layer network using chained LazyLinear modules.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two lazy linear layers with output
                # features 15 and 10.
                super().__init__()
                self.linear_1 = torch.nn.LazyLinear(15)
                self.linear_2 = torch.nn.LazyLinear(10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first layer then the second layer and
                # returns the result.
                y = self.linear_1(x)
                return self.linear_2(y)

        net = MyNetwork()
        net(torch.ones(5, 10))
        self.assertTrue(net.linear_1.weight.shape == (15, 10))
        self.assertTrue(net.linear_1.bias.shape == (15,))
        self.assertTrue(net.linear_2.weight.shape == (10, 15))
        self.assertTrue(net.linear_2.bias.shape == (10,))

    @suppress_warnings
    def test_optimizer_pass(self):
        # [FUNCTION DOC] For each optimizer class in a fixed list, constructs it over a module
        # with an uninitialized parameter, materializes the parameter, checks the optimizer's
        # param group now holds a real Parameter, sets random gradients and steps (LBFGS gets a
        # closure). SGD uses learning rate zero; Adagrad is expected to raise a ValueError
        # about uninitialized parameters at construction and is skipped.
        optimizers = [
            torch.optim.Adadelta,
            torch.optim.Adagrad,
            torch.optim.Adamax,
            torch.optim.Adam,
            torch.optim.AdamW,
            torch.optim.ASGD,
            torch.optim.SGD,
            torch.optim.Rprop,
            torch.optim.RMSprop,
            torch.optim.LBFGS,
            torch.optim.NAdam,
            torch.optim.RAdam,
        ]

        def run_step(module, optim):
            # [FUNCTION DOC] HELPER: Verifies the first param group entry is uninitialized,
            # materializes it with 10 elements, verifies the entry became a normal Parameter,
            # assigns random gradients to all parameters and runs one optimizer step.
            self.assertIsInstance(
                optim.param_groups[0]["params"][0], UninitializedParameter
            )
            module.test_param.materialize(10)
            self.assertIsInstance(optim.param_groups[0]["params"][0], Parameter)
            self.assertNotIsInstance(
                optim.param_groups[0]["params"][0], UninitializedParameter
            )
            for p in module.parameters():
                p.grad = torch.rand_like(p)
            if isinstance(optim, torch.optim.LBFGS):
                optim.step(lambda: 1.0)
            else:
                optim.step()

        for optim_cls in optimizers:
            module = LazyModule()
            module.register_parameter("test_param", UninitializedParameter())
            if optim_cls is torch.optim.SGD:
                optim = optim_cls(module.parameters(), lr=0.0)
            elif optim_cls is torch.optim.Adagrad:
                with self.assertRaisesRegex(ValueError, "uninitialized parameter"):
                    optim = optim_cls(module.parameters())
                continue
            else:
                optim = optim_cls(module.parameters())
            run_step(module, optim)

    @suppress_warnings
    def test_weight_norm(self):
        # [FUNCTION DOC] Asserts weight_norm on a LazyLinear raises a ValueError saying the
        # module has uninitialized parameters.
        m = nn.LazyLinear(7)
        with self.assertRaisesRegex(ValueError, "have uninitialized parameters."):
            m = torch.nn.utils.weight_norm(m)

    @suppress_warnings
    def test_spectral_norm(self):
        # [FUNCTION DOC] Asserts spectral_norm on a LazyLinear raises a ValueError saying the
        # module has uninitialized parameters.
        m = nn.LazyLinear(7)
        with self.assertRaisesRegex(ValueError, "have uninitialized parameters."):
            m = torch.nn.utils.spectral_norm(m)

    @suppress_warnings
    def test_invalid_functions(self):
        # [FUNCTION DOC] Asserts that torch.empty_like, torch.add and the plus operator applied
        # to an uninitialized parameter each raise a ValueError about an uninitialized
        # parameter.
        param = torch.nn.parameter.UninitializedParameter()
        with self.assertRaisesRegex(ValueError, "uninitialized parameter"):
            torch.empty_like(param)

        with self.assertRaisesRegex(ValueError, "uninitialized parameter"):
            torch.add(param, param)

        with self.assertRaisesRegex(ValueError, "uninitialized parameter"):
            param + param


class TestLazyModulesDevice(NNTestCase):
    # [CLASS DOC] Accelerator-classified test case (derived from NNTestCase) for lazy module
    # materialization on devices; instantiated per device type at import time.
    hw_classification = HardwareClassification.ACCELERATOR

    @suppress_warnings
    def test_materialize_device(self, device):
        # [FUNCTION DOC] Checks an uninitialized parameter materializes on CPU by default, and
        # after the module is moved to the given device it materializes on that device type.
        # Takes the device string supplied by the device-type test instantiation.
        module = LazyModule()
        module.register_parameter("test_param", UninitializedParameter())
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.device.type == "cpu")

        # Test materialization on the current device
        module = LazyModule()
        module.register_parameter("test_param", UninitializedParameter())
        module.to(device)
        module.test_param.materialize(10)
        self.assertEqual(module.test_param.device.type, device.split(":")[0])


instantiate_device_type_tests(TestLazyModulesDevice, globals())


if __name__ == "__main__":
    run_tests()
