# Owner(s): ["module: vmap"]
# ruff: noqa: F841
# [MODULE DOC] Legacy (torch._vmap_internals) vmap tests. Defines a reference vmap that loops
# and stacks, a TensorFactory of random inputs, a comparison helper, and a base TestCase (nested in
# Namespace) that fails tests which hit the slow vmap fallback unless explicitly allowed.
# Test classes cover the vmap API, many operators, and batched autograd gradients; the last is
# instantiated per device type at import time.

import functools
import itertools
import types
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch._vmap_internals import vmap
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


FALLBACK_REGEX = r"There is a performance drop"


class EnableVmapFallbackWarnings:
    # [CLASS DOC] Context manager that turns on the debug-only vmap fallback performance warnings.
    # Entering saves the current enabled state on the instance (prev_state) and enables the
    # warnings via the C++ debug switch; exiting restores the saved state. Used by tests to detect
    # use of the slow vmap fallback path.
    def __enter__(self):
        # [FUNCTION DOC] HELPER: Saves whether fallback warnings are currently enabled into
        # prev_state, then enables them. Takes no arguments and returns nothing.
        self.prev_state = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
        torch._C._debug_only_display_vmap_fallback_warnings(True)

    def __exit__(self, *ignored):
        # [FUNCTION DOC] HELPER: Restores the fallback-warning state saved on entry. Extra
        # exception arguments are ignored and exceptions are never suppressed (returns nothing).
        torch._C._debug_only_display_vmap_fallback_warnings(self.prev_state)


class TestVmapAPILegacy(TestCase):
    # [CLASS DOC] Tests for the legacy torch._vmap_internals vmap API surface: output type
    # validation, in_dims / out_dims handling and their error messages, nesting, the fallback
    # warning behavior, and unsupported autograd interactions. All tests run on CPU tensors and
    # use no shared state besides module helpers.
    def test_non_tensor_output_raises(self):
        # [FUNCTION DOC] Checks that vmap raises ValueError when a function returns a non-Tensor:
        # a bare float, and a tuple whose second element is an int (error names the offending
        # return index).
        with self.assertRaisesRegex(
            ValueError, "got type <class 'float'> as the return"
        ):
            output = vmap(lambda x: 3.14)(torch.ones(3))

        def multiple_outputs(x):
            # [FUNCTION DOC] HELPER: Returns a tuple of the input and the integer 3, i.e. an
            # invalid non-Tensor second output.
            return x, 3

        with self.assertRaisesRegex(ValueError, "got type <class 'int'> for return 1"):
            vmap(multiple_outputs)(torch.ones(3))

    def test_different_map_dim_size_raises(self):
        # [FUNCTION DOC] Verifies ValueError about mismatched mapped-dimension sizes (sizes 2 vs
        # 3) for plain positional inputs, a tuple input with tuple in_dims, and a dict input with
        # dict in_dims.
        x = torch.randn(2)
        y = torch.randn(3)
        expected_msg = (
            "Expected all tensors to have the same size in the mapped dimension"
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(torch.mul)(x, y)
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z["x"] + z["y"], in_dims=({"x": 0, "y": 0},))(
                {"x": x, "y": y}
            )

    def test_func_with_no_inputs(self):
        # [FUNCTION DOC] Verifies that vmap over a function called with no arguments, whether the
        # function takes none or one parameter, raises ValueError saying it got no inputs.
        expected_msg = "got no inputs"

        def foo():
            # [FUNCTION DOC] HELPER: Zero-parameter function returning a random 3-element tensor.
            return torch.randn(3)

        def bar(x):
            # [FUNCTION DOC] HELPER: One-parameter function ignoring its input and returning a
            # random 3-element tensor.
            return torch.randn(3)

        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(foo)()

        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(bar)()

    def test_constant_function(self):
        # [FUNCTION DOC] Checks that a function returning a constant scalar tensor, ignoring the
        # batched input of size 3, produces a result of that constant repeated once per batch
        # entry.
        output = vmap(lambda x: torch.tensor(3.14))(torch.ones(3))
        self.assertEqual(output, torch.tensor([3.14, 3.14, 3.14]))

    def test_single_input(self):
        # [FUNCTION DOC] Checks vmap of a single-input elementwise square equals squaring the
        # whole tensor.
        x = torch.randn(2, 3)

        def square(x):
            # [FUNCTION DOC] HELPER: Returns the elementwise square of its input.
            return x * x

        output = vmap(square)(x)
        self.assertEqual(output, x * x)

    def test_multiple_inputs(self):
        # [FUNCTION DOC] Checks vmap of torch.mul over two same-shaped inputs equals the direct
        # product.
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        output = vmap(torch.mul)(x, y)
        self.assertEqual(output, x * y)

    def test_multiple_outputs(self):
        # [FUNCTION DOC] Checks a function returning a tuple of two tensors yields a tuple whose
        # entries equal the square and cube of the input.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the pair (square, cube) of the input.
            return x * x, x * x * x

        x = torch.randn(3)
        outputs = vmap(foo)(x)
        self.assertEqual(outputs[0], x * x)
        self.assertEqual(outputs[1], x * x * x)

    def test_multiple_outputs_error_cases(self):
        # [FUNCTION DOC] Checks that returning a tuple of tensors works, while returning a list
        # (of one or two tensors) raises ValueError saying outputs must only be Tensors, got a
        # list.
        # This is the same thing as
        # def returns_tuple_of_tensors(x):
        #     return x, x
        def returns_tuple_of_tensors(x):
            # [FUNCTION DOC] HELPER: Returns the input twice as a tuple.
            return (x, x)

        def returns_list_of_two_tensors(x):
            # [FUNCTION DOC] HELPER: Returns a two-element list containing the input twice.
            return [x, x]

        def returns_list_of_one_tensor(x):
            # [FUNCTION DOC] HELPER: Returns a one-element list containing the input.
            return [x]

        x = torch.randn(3)

        # should not throw
        vmap(returns_tuple_of_tensors)(x)

        # jax supports these, but we don't yet
        msg = "must only return Tensors, got type <class 'list'>"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(returns_list_of_two_tensors)(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(returns_list_of_one_tensor)(x)

    def test_nested_with_same_map_dim(self):
        # [FUNCTION DOC] Checks nested vmap (two and three levels) of torch.mul over inputs with
        # the same leading dims matches the plain product.
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        output = vmap(vmap(torch.mul))(x, y)
        self.assertEqual(output, x * y)

        output = vmap(vmap(vmap(torch.mul)))(x, y)
        self.assertEqual(output, x * y)

    def test_nested_with_different_map_dim(self):
        # [FUNCTION DOC] Checks nested vmaps mapping over different tensors (different sizes) give
        # outer product style results of shape batch1 x batch2 (x batch3) x 3, for two and three
        # levels.
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        output = vmap(lambda x: vmap(lambda y: x * y)(y))(x)
        self.assertEqual(output.shape, (2, 5, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        z = torch.randn(7, 3)
        output = vmap(lambda x: vmap(lambda y: vmap(lambda z: x * y * z)(z))(y))(x)
        self.assertEqual(output.shape, (2, 5, 7, 3))
        self.assertEqual(output, x.view(2, 1, 1, 3) * y.view(5, 1, 3) * z)

    def test_noop_in_inner_vmap(self):
        # [FUNCTION DOC] Checks that an inner vmap whose function returns the outer-mapped value
        # unchanged broadcasts it along the inner batch dimension (expanded result).
        x = torch.randn(3)
        y = torch.randn(5)
        output = vmap(lambda x: vmap(lambda y: x)(y))(x)
        self.assertEqual(output, x.view(3, 1).expand(3, 5))

    def test_unsupported_op_err_msg(self):
        # [FUNCTION DOC] Verifies RuntimeError "batching rule not implemented" for ops the
        # fallback cannot handle: a view op (ravel), an out= variant, a tensor-list argument, and
        # an op returning a non-tensor (item).
        # Unsupported view op
        tensor = torch.randn(2, 3)
        msg = (
            r"Batching rule not implemented for aten::.+; the "
            r"fallback path doesn't work on out= or view ops"
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(torch.ravel)(tensor)

        def out_op(x, y):
            # [FUNCTION DOC] HELPER: Calls abs on x writing into the out tensor y and returns it.
            return torch.abs(x, out=y)

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(out_op)(tensor, tensor)

        tensor = torch.randn(2)
        # The fallback doesn't support TensorList
        with self.assertRaisesRegex(RuntimeError, "Batching rule not implemented"):
            vmap(lambda t: torch.atleast_1d([t]))(tensor)

        # Don't support non-tensor returns. This is a limitation of vmap;
        # functions that don't return tensors must be special cased
        with self.assertRaisesRegex(RuntimeError, "Batching rule not implemented"):
            vmap(torch.Tensor.item)(tensor)

    def test_nonzero_out_dims(self):
        # [FUNCTION DOC] Checks out_dims for identity functions: the batch dim lands at the
        # requested position (positive, negative, applied to all outputs, and for a 64-dim
        # tensor), the result shares storage with the input (same data pointer), and a non-
        # identity multi-output function.
        # Basic test
        tensor = torch.randn(2, 3)
        result = vmap(lambda x: x, out_dims=1)(tensor)
        self.assertEqual(result, tensor.permute(1, 0))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # Test that the batch dimension gets permuted to dim 2
        tensor = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x: x, out_dims=2)(tensor)
        self.assertEqual(result, tensor.permute(1, 2, 0, 3))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # negative out_dim
        tensor = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x: x, out_dims=-1)(tensor)
        self.assertEqual(result, tensor.permute(1, 2, 3, 0))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # check that out_dims works on ALL outputs
        tensor = torch.randn(2, 3, 5, 7)
        other = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x, y: (x, y), out_dims=2)(tensor, other)
        self.assertEqual(
            result, (tensor.permute(1, 2, 0, 3), other.permute(1, 2, 0, 3))
        )

        # use out_dims with the maximum vmap-able tensor dims (64 dims)
        ndims = 64
        shape = [2] + [1] * (ndims - 1)
        expected_shape = [1, 1, 2] + [1] * (ndims - 3)
        tensor = torch.randn(shape)
        result = vmap(lambda x: x, out_dims=2)(tensor)
        self.assertEqual(result.shape, expected_shape)

        # test something that is not the identity function
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns the triple of x, x times y, and x times y squared.
            return x, x * y, x * y * y

        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        result = vmap(foo, out_dims=1)(x, y)
        self.assertEqual(
            result,
            (
                x.permute(1, 0, 2),
                (x * y).permute(1, 0, 2),
                (x * y * y).permute(1, 0, 2),
            ),
        )

    def test_multiple_out_dims(self):
        # [FUNCTION DOC] Checks out_dims given as a tuple with one entry per output, including
        # negative entries, for functions with two and four outputs.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the input twice as a tuple.
            return x, x

        def bar(x, y):
            # [FUNCTION DOC] HELPER: Returns a four-tuple of x, x, x and the product of x and y.
            return x, x, x, x * y

        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        result = vmap(foo, out_dims=(0, 1))(x)
        self.assertEqual(result, (x, x.permute(1, 0, 2)))

        result = vmap(bar, out_dims=(-1, 0, 1, 2))(x, y)
        expected = (
            x.permute(1, 2, 0),
            x,
            x.permute(1, 0, 2),
            (x * y).permute(1, 2, 0),
        )
        self.assertEqual(result, expected)

    def test_nested_out_dims(self):
        # [FUNCTION DOC] Checks out_dims for nested vmaps: inner only, outer and inner, negative
        # values, and a non-identity inner function; verifies result shape and equality to
        # permuted references.
        y = torch.randn(2, 3, 5, 7)

        # Inner vmap has non-zero out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y))(y)
        self.assertEqual(result.shape, (2, 5, 3, 7))
        self.assertEqual(result, y.permute(0, 2, 1, 3))

        # all vmaps have non-zero out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y), out_dims=1)(y)
        self.assertEqual(result.shape, (5, 2, 3, 7))
        self.assertEqual(result, y.permute(2, 0, 1, 3))

        # throwing in some negative out_dims
        result = vmap(lambda y: vmap(lambda x: x, out_dims=-1)(y), out_dims=-1)(y)
        self.assertEqual(result.shape, (5, 7, 3, 2))
        self.assertEqual(result, y.permute(2, 3, 1, 0))

        # testing fn that isn't the identity
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        result = vmap(lambda y: vmap(lambda x: x * y, out_dims=1)(x), out_dims=-1)(y)
        self.assertEqual(result.shape, (3, 2, 5))
        self.assertEqual(result, (y.view(5, 1, 3) * x).permute(2, 1, 0))

    def test_out_dims_edge_case(self):
        # [FUNCTION DOC] Checks that out_dims given as a one-element tuple is accepted for a
        # single output function and equals the integer form.
        def foo(x):
            # [FUNCTION DOC] HELPER: Identity function.
            return x

        # Test that we accept out_dims=(1,) for a function with one output.
        tensor = torch.randn(2, 3)
        expected = vmap(foo, out_dims=1)(tensor)
        result = vmap(foo, out_dims=(1,))(tensor)
        self.assertEqual(result, expected)

    def test_out_dims_must_be_int_or_tuple_of_int_err_msg(self):
        # [FUNCTION DOC] Checks ValueError with the message that out_dims must be an int or tuple
        # of int, for a string, a tuple holding a string, None, and a tuple holding None.
        msg = "`out_dims` must be an int or a tuple of int"
        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims="lol")(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=("lol",))(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=None)(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=(None,))(tensor)

    def test_out_dims_and_num_outputs_mismatch_err_msg(self):
        # [FUNCTION DOC] Checks ValueError when the number of out_dims entries differs from the
        # number of outputs, both too many and too few.
        msg = "`out_dims` must have one dim per output"
        x = torch.randn(2, 3, 5)

        # Too many out_dims
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=(0, 0))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0, 0, 0))(x)

        # Too few out_dims
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x), out_dims=(0,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0))(x)

    def test_out_dim_out_of_bounds_err_msg(self):
        # [FUNCTION DOC] Checks that out_dims outside the valid range (3 and -4 for a result
        # needing dims 0..2) raise IndexError with a dimension-out-of-range message.
        # TODO(rzou): This error message isn't that great. It comes straight
        # from maybe_wrap_dim. Consider doing a try-catch-(add some context) to
        # the error message in the future in C++
        msg = "Dimension out of range"
        x = torch.randn(2, 3, 5)
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=3)(x)
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=-4)(x)

    def test_non_zero_in_dims(self):
        # [FUNCTION DOC] Checks non-zero in_dims: a single input mapped over dim 1 moves the batch
        # dim to the front without copying, and two-input mappings with swapped dims match
        # transposes.
        tensor = torch.randn(2, 3, 5)

        # Implicit out_dims = 0; vmap will move the batch dim to the front.
        output = vmap(lambda x: x, (1,))(tensor)
        self.assertEqual(output, tensor.permute(1, 0, 2))
        self.assertEqual(output.data_ptr(), tensor.data_ptr())

        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        output = vmap(torch.mul, (0, 1))(x, y)
        self.assertEqual(output, x * y.t())
        output = vmap(torch.mul, (1, 0))(x, y)
        self.assertEqual(output, x.t() * y)

    def test_none_in_dims(self):
        # [FUNCTION DOC] Checks that an in_dim of None leaves a tensor unmapped (broadcast against
        # the mapped input) and also works for a non-tensor argument.
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # None in_dim for a Tensor means we don't map over it
        output = vmap(torch.mul, (0, None))(x, y)
        self.assertEqual(output.shape, (2, 2, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        # None in_dim for non-tensor arguments
        output = vmap(torch.mul, (0, None))(x, 2)
        self.assertEqual(output, x * 2)

    def test_nested_non_default_in_dims(self):
        # [FUNCTION DOC] Checks triple-nested vmap with different in_dims at each level matches a
        # product of suitably permuted inputs.
        x = torch.rand(5, 2, 3)
        y = torch.rand(3, 5, 2)
        result = vmap(vmap(vmap(torch.mul), (1, 0)), (1, 2))(x, y)
        self.assertEqual(result, x.permute(1, 2, 0) * y.permute(2, 0, 1))

    def test_non_default_in_dims_out_dims(self):
        # [FUNCTION DOC] Checks combinations of in_dims and out_dims, for identity and for an
        # elementwise function, including nesting; also checks storage is shared for identity
        # cases.
        x = torch.randn(2, 3, 5)

        # Same in_dim as out_dim, vmap over identity
        result = vmap(lambda x: x, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x)
        self.assertEqual(result.data_ptr(), x.data_ptr())

        # Different in_dim from out_dim, vmap over identity
        result = vmap(lambda x: x, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, x.transpose(1, 2))
        self.assertEqual(result.data_ptr(), x.data_ptr())

        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the input multiplied by two.
            return x * 2

        # Same in_dim as out_dim, vmap over operation
        result = vmap(foo, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x * 2)

        # Different in_dim as out_dim, vmap over operation
        result = vmap(foo, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, (x * 2).transpose(1, 2))

        # Basic nested test.
        result = vmap(vmap(foo, 1, 1), 1, 1)(x)
        self.assertEqual(result, x * 2)

    def test_accepts_nested_inputs(self):
        # [FUNCTION DOC] Checks vmap accepts pytree inputs (tuple, list, dict, deeply nested
        # mixes) with matching or default in_dims, producing the sum of components.
        B0 = 2
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # Single layer of nesting
        out = vmap(lambda z: z[0] + z[1])((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        self.assertEqual(out, x + y)

        out = vmap(lambda z: z[0] + z[1])([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, y])
        self.assertEqual(out, x + y)

        out = vmap(lambda z: z["x"] + z["y"])({"x": x, "y": y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z["x"] + z["y"], in_dims=(0,))({"x": x, "y": y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z["x"] + z["y"], in_dims=({"x": 0, "y": 0},))(
            {"x": x, "y": y}
        )
        self.assertEqual(out, x + y)

        # Multiple layers of nesting
        out_fn = vmap(lambda z: z["x"][0] + z["x"][1][0] + z["y"][0] + z["y"][1])
        out = out_fn({"x": [x, (x,)], "y": [y, y]})
        self.assertEqual(out, x + x + y + y)

    def test_in_dims_wrong_type_err_msg(self):
        # [FUNCTION DOC] Checks that in_dims given as a list, set or string, rather than an int or
        # tuple, raises ValueError about expecting an int or nested tuple; a tuple works.
        x = torch.randn(3)
        y = torch.randn(3)
        msg = r"expected `in_dims` to be int or a \(potentially nested\) tuple"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, [0, 0])(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, set({0}))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, "lol")(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=[0, 0])([x, y])
        # The following should not throw
        vmap(torch.mul, (0, 0))(x, y)

    def test_not_enough_in_dims_err_msg(self):
        # [FUNCTION DOC] Checks ValueError when in_dims has the wrong structure (too few or too
        # many entries, or wrong nesting) relative to the inputs; matching structure works.
        x = torch.randn(3)
        y = torch.randn(3)
        msg = r"in_dims is not compatible with the structure of `inputs`"

        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0,))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0, 0, 0))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0],))([x, y])
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))([x, y])
        # The following should not throw
        vmap(torch.mul, (0, 0))(x, y)

    def test_integer_in_dim_but_not_tensor_input_err_msg(self):
        # [FUNCTION DOC] Checks ValueError when an integer in_dim is paired with an input that is
        # not a Tensor (for example the int 0 or 1), while None is accepted for such inputs.
        def foo(xy):
            # [FUNCTION DOC] HELPER: Unused; returns the product of the two elements of its
            # argument.
            return xy[0] * xy[1]

        def bar(x, yz):
            # [FUNCTION DOC] HELPER: Unused; returns the product of x and the two elements of yz.
            return x * yz[0] * yz[1]

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # the following are errors in jax (and will always be errors)
        msg = "Got in_dim=0 for an input but the input is of type"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum)(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum, (0, 0))(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, 1])
        # The following should not throw
        vmap(torch.sum, (0, None))(x, 0)

    def test_in_dim_not_in_tensor_err_msg(self):
        # [FUNCTION DOC] Checks ValueError when an in_dim is not valid for the tensor's
        # dimensionality (scalar tensor, out of range positive or negative dims); valid dims do
        # not raise.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the elementwise square of its input.
            return x * x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        msg = r"Got in_dim=-?\w for some input, but that input is a Tensor of dimensionality \w"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo)(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(0,))(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(-1,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(2,))(y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([3, 0],))([x, y])
        # the following should not throw
        vmap(foo, in_dims=(0,))(torch.randn(2, 3))
        vmap(foo, in_dims=(1,))(torch.randn(2, 3))

    def test_fallback_does_not_warn_by_default(self):
        # [FUNCTION DOC] Checks that vmapping atan2, which uses the fallback, emits only the
        # single experimental-vmap warning (one warning in total) when fallback warnings are not
        # enabled.
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.atan2
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            result = vmap(op)(x, y)
            # The single warning here is the "vmap is experimental"
            # warning, not a warning from the vmap fallback path.
            self.assertEqual(len(wa), 1)

    def test_fallback_warns_when_warnings_are_enabled(self):
        # [FUNCTION DOC] Checks that with fallback warnings enabled, vmapping atan2 emits two
        # warnings and the last matches the performance-drop text.
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.atan2
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            with EnableVmapFallbackWarnings():
                result = vmap(op)(x, y)
            self.assertEqual(len(wa), 2)
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def _assert_uses_vmap_fallback(self, vmap_args, inputs):
        # [FUNCTION DOC] HELPER: Runs vmap with the given argument tuple on the given inputs with
        # fallback warnings enabled and asserts exactly two warnings occurred, the last matching
        # the performance-drop regex.
        with warnings.catch_warnings(record=True) as wa:
            with EnableVmapFallbackWarnings():
                result = vmap(*vmap_args)(*inputs)
            self.assertEqual(len(wa), 2)
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def test_fallback_zero_dim(self):
        # [FUNCTION DOC] Checks the fallback refuses to vmap over a dimension of size 0
        # (RuntimeError) for atan2, for one or both args mapped and for nested batch dims.
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.atan2
        x = torch.randn(11)
        y = torch.randn(11)
        self._assert_uses_vmap_fallback((op,), (x, y))

        B0, B1 = 0, 3
        x = torch.randn(B0, 11)
        y = torch.randn(11)

        msg = "The fallback path does not support vmap over dims of size 0"

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)

        x = torch.randn(B0, B1, 11)
        y = torch.randn(B1, 11)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)

    def test_fallback_atan2(self):
        # [FUNCTION DOC] Checks correctness of the fallback for atan2 against a direct call:
        # simple, permuted in_dims, nested vmap and a large total batch size.
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.atan2

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)

        self._assert_uses_vmap_fallback((op,), (x, y))

        # fallback on torch.atan2
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(op, (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # fallback on torch.atan2, nested vmap
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # big batch size (total 10000)
        x = torch.randn(100, 10, 10, 5)
        y = torch.randn(100, 10, 10)
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(result, op(x, y.view(100, 10, 10, 1)))

    def test_fallback_masked_fill(self):
        # [FUNCTION DOC] Checks the fallback for index_add (despite the name) with a mapped self
        # and values and unmapped dim and index, for batch sizes 5 and 1237.
        # NB: One day we will implement a batching rule for masked_fill
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        def run_test(batch_size):
            # [FUNCTION DOC] HELPER: For one batch size, asserts fallback usage, then compares
            # vmap of index_add against calling it on the unbatched layout.
            B0 = batch_size
            x = torch.randn(B0, 7, 11, 13)
            dim = 0
            index = torch.tensor([0, 4, 2])
            values = torch.randn(B0, 3, 11, 13)

            self._assert_uses_vmap_fallback(
                (torch.index_add, (0, None, None, 0)), (x, dim, index, values)
            )

            result = vmap(torch.index_add, (0, None, None, 0))(x, dim, index, values)
            expected = torch.index_add(x, dim + 1, index, values.view(B0, 3, 11, 13))
            self.assertEqual(result, expected)

        run_test(batch_size=5)
        run_test(batch_size=1237)

    def test_fallback_multiple_returns(self):
        # [FUNCTION DOC] Checks the fallback for var_mean, an op with multiple returns, for
        # single, nested and large-batch vmap against the direct result along the matching last
        # dim.
        # NB: One day we will implement a batching rule for torch.var_mean
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        B0, B1, B2 = 2, 3, 1237
        tensor = torch.randn(B0, 10)

        self._assert_uses_vmap_fallback((torch.var_mean,), (tensor,))

        # fallback correctness on torch.var_mean
        result = vmap(torch.var_mean)(tensor)
        expected = torch.var_mean(tensor, dim=1)
        self.assertEqual(result, expected)

        # nested vmap
        tensor = torch.randn(B0, B1, 10)
        result = vmap(vmap(torch.var_mean))(tensor)
        expected = torch.var_mean(tensor, dim=2)
        self.assertEqual(result, expected)

        # big batch size, nested vmap
        tensor = torch.randn(B0, B1, B2, 10)
        result = vmap(vmap(vmap(torch.var_mean)))(tensor)
        expected = torch.var_mean(tensor, dim=3)
        self.assertEqual(result, expected)

    def test_inplace_fallback_unary(self):
        # [FUNCTION DOC] Checks the in-place fallback for acos_: the result is the same object as
        # the input and holds acos values; with out_dims one the result is a view of it; also
        # nested and large-batch cases.
        # Test the in-place fallback on an in-place method that takes no
        # additional Tensor arguments. This is the simplest case of the fallback.
        # NB: One day we will implement a batching rule for acos_.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.acos_
        B0, B1, B2 = 2, 3, 10000

        x = torch.randn(B0, 5)
        self._assert_uses_vmap_fallback((op,), (x,))

        # Single vmap
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        result = vmap(op)(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

        # Single vmap + different out_dim produces a view(!)
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        result = vmap(op, out_dims=(1,))(x)
        self.assertTrue(result._base is x)
        self.assertEqual(result, x_orig.t().acos())

        # Nested vmap
        x_orig = torch.randn(B0, B1, 5)
        x = x_orig.clone()
        result = vmap(vmap(op))(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

        # Nested vmap, large batch size
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        result = vmap(vmap(vmap(op)))(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

    def test_inplace_fallback_nary_same_levels(self):
        # [FUNCTION DOC] Checks the in-place fallback for atan2_ when both operands have the same
        # batch levels (single, nested, and large batch); the mutated input equals the out-of-
        # place result.
        # NB: One day we will implement a batching rule for atan2_
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.atan2_
        outplace_op = torch.atan2

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)
        self._assert_uses_vmap_fallback((op,), (x, y))

        # Single vmap
        B0 = 5
        x_orig = torch.randn(7, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, 7, 11)
        vmap(op, (2, 0))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.movedim(0, 2)))

        # Nested vmap
        B0, B1 = 5, 7
        x_orig = torch.randn(B1, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, B1, 11)
        vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.movedim([0, 1], [2, 0])))

        # big batch size (total 10000)
        B0, B1, B2 = 100, 10, 10
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        y = torch.randn(B0, B1, B2)
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, B1, B2, 1)))

    def test_inplace_fallback_nary_different_levels(self):
        # [FUNCTION DOC] Checks in-place fallback for atan2_ when operand batch levels differ:
        # allowed if every level of the other operand appears in self; otherwise RuntimeError
        # saying the vmap in-place op is not possible.
        # NB: One day we will implement a batching rule for atan2_
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.atan2_
        outplace_op = torch.atan2
        B0, B1, B2 = 2, 3, 5

        x = torch.rand(B0, 7)
        y = torch.rand(7)
        self._assert_uses_vmap_fallback((op, (0, None)), (x, y))

        # op(left, right): All of the levels in right are found in left
        x_orig = torch.rand(B0, 7)
        x = x_orig.clone()
        y = torch.rand(7)
        vmap(op, in_dims=(0, None))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y))

        x_orig = torch.rand(B0, B1, 7)
        x = x_orig.clone()
        y = torch.rand(B0, 7)
        vmap(vmap(op, in_dims=(0, None)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, 1, 7)))

        # op(left, right): Some of the levels in right are not found in left
        msg = r"vmap: aten::atan2_\(self, \*extra_args\) is not possible"
        x = torch.rand(7)
        y = torch.rand(B0, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(x, y)

        x = torch.rand(B1, 7)
        y = torch.rand(B0, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 0))(x, y)

        x = torch.rand(B1, 7)
        y = torch.rand(7, B0)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 1))(x, y)

        x = torch.rand(B0, 7)
        y = torch.rand(B0, B1, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(None, 0)))(x, y)

    def test_backward_unsupported_interaction(self):
        # [FUNCTION DOC] Checks calling backward inside vmap raises RuntimeError, whether the
        # vmapped tensor, a vmapped gradient, or an unrelated captured tensor is involved.
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(5)
        grad = torch.randn_like(x)
        err_msg = r"backward\(\) called inside torch.vmap"

        def backward_on_vmapped_tensor(x):
            # [FUNCTION DOC] HELPER: Calls backward on the sum of x.
            x.sum().backward()

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_on_vmapped_tensor)(x)

        def backward_with_vmapped_grad(x, grad):
            # [FUNCTION DOC] HELPER: Calls backward on x with the given gradient.
            x.backward(grad)

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_with_vmapped_grad)(x, grad)

        def completely_unrelated_backward(y):
            # [FUNCTION DOC] HELPER: Ignores y and calls backward on the sum of the captured outer
            # tensor x.
            x.sum().backward()

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(completely_unrelated_backward)(y)

    def test_grad_unsupported_interaction(self):
        # [FUNCTION DOC] Checks autograd.grad inside vmap raises RuntimeError when the gradient
        # output or input is a vmapped tensor.
        input_tensor = torch.randn(3, requires_grad=True)
        err_msg = "autograd.grad.* called inside torch.vmap"

        captured = torch.randn(3, requires_grad=True)

        def output_to_grad_is_vmapped(input_tensor):
            # [FUNCTION DOC] HELPER: Builds an output from a captured tensor and the vmapped input
            # and returns the gradient with respect to the captured tensor.
            output = (captured * input_tensor).sum()
            return torch.autograd.grad([output], [captured])[0]

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(output_to_grad_is_vmapped)(input_tensor)

        output = (input_tensor**2).sum()

        def input_to_grad_is_vmapped(input_tensor):
            # [FUNCTION DOC] HELPER: Returns the gradient of the outer-defined output with respect
            # to the vmapped input.
            return torch.autograd.grad([output], [input_tensor])[0]

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(input_to_grad_is_vmapped)(input_tensor)

    def test_batched_gradient_basic(self):
        # [FUNCTION DOC] Checks that batched vector-Jacobian products via vmap over an identity
        # matrix of grad outputs produce the Jacobian (diagonal matrix of y) for an elementwise
        # product.
        N = 3
        x = torch.randn(N, requires_grad=True)
        y = torch.randn(N)

        def vjp_mul(v):
            # [FUNCTION DOC] HELPER: Returns the vector-Jacobian product of x times y with respect
            # to x for the cotangent v.
            return torch.autograd.grad([x * y], [x], grad_outputs=[v])[0]

        batched_v = torch.eye(N)
        jacobian = vmap(vjp_mul)(batched_v)
        self.assertEqual(jacobian, torch.diagflat(y))

    def test_functools_partial(self):
        # [FUNCTION DOC] Checks vmap works with a functools.partial callable.
        x = torch.randn(3)
        y = torch.randn(2, 3)
        result = vmap(functools.partial(torch.mul, x))(y)
        self.assertEqual(result, x * y)

    def test_nn_module(self):
        # [FUNCTION DOC] Checks vmap works with an nn.Module (Linear without bias) as the
        # function.
        tensor = torch.randn(2, 3)
        model = torch.nn.Linear(3, 3, bias=False)
        result = vmap(model)(tensor)
        self.assertEqual(result, model(tensor))

    def test_fallback_with_undefined_grad(self):
        # [FUNCTION DOC] Checks the fallback copes with an undefined gradient (conv2d backward
        # without a bias) by vmapping a vjp function without error while using the fallback.
        B0 = 7
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        weight = torch.randn(3, 3, 1, 1)
        v = torch.randn(B0, 2, 3, 4, 5)

        def get_vjp(v):
            # [FUNCTION DOC] HELPER: Returns the input gradient of conv2d with the captured input
            # and weight, for the cotangent v.
            result = torch.nn.functional.conv2d(x, weight)
            (grad_x,) = torch.autograd.grad(result, x, v)
            return grad_x

        # Runs vmap(get_vjp)(v), which should not error out.
        # The backward formula for convolution returns an undefined
        # Tensor for grad_bias because the original bias does not exist.
        #
        # In the future we'll probably add a batching rule for convolution
        # backward. When this happens, we should modify this test to use a
        # different op (and/or create and use a dummy operator) to avoid bitrot.
        self._assert_uses_vmap_fallback([get_vjp], [v])


def slice_inputs(inputs, bdims, i):
    # [FUNCTION DOC] HELPER: For each input and its batch dim, selects index i along that dim, or
    # keeps the input whole if the dim is None. Returns a tuple of the results.
    result = []
    for inp, bdim in zip(inputs, bdims):
        if bdim is None:
            result.append(inp)
        else:
            result.append(inp.select(bdim, i))
    return tuple(result)


def reference_vmap(op, inputs, in_dims=0, out_dims=0):
    # [FUNCTION DOC] HELPER: Slow reference implementation of vmap: slices inputs along in_dims,
    # calls op per index, and stacks results with torch.stack along out_dims (int broadcast to all
    # returns). Handles single and tuple returns. Raises AssertionError if batch sizes differ,
    # there are no results, or results are not uniformly tensors or equal-length tuples.
    if isinstance(in_dims, int):
        in_dims = (in_dims,) * len(inputs)
    bdim_sizes = [inp.size(dim) for inp, dim in zip(inputs, in_dims) if dim is not None]
    if not all(bdim_size == bdim_sizes[0] for bdim_size in bdim_sizes):
        raise AssertionError("all batch dimensions must have the same size")
    bdim_size = bdim_sizes[0]
    results = tuple(op(*slice_inputs(inputs, in_dims, i)) for i in range(bdim_size))

    if len(results) == 0:
        raise AssertionError("results must not be empty")
    op_has_single_return = not isinstance(results[0], tuple)
    if op_has_single_return:
        if not all(isinstance(result, torch.Tensor) for result in results):
            raise AssertionError("all results must be tensors")
        if isinstance(out_dims, int):
            out_dims = (out_dims,) * 1
        return torch.stack(results, dim=out_dims[0])

    if not all(isinstance(result, tuple) for result in results):
        raise AssertionError("all results must be tuples")
    num_returns = len(results[0])
    if not all(len(result) == num_returns for result in results):
        raise AssertionError("all results must have the same number of returns")
    if isinstance(out_dims, int):
        out_dims = (out_dims,) * num_returns
    return tuple(
        torch.stack(result_shards, out_dim)
        for result_shards, out_dim in zip(zip(*results), out_dims)
    )


class TensorFactory:
    # [CLASS DOC] Namespace of static factories producing CPU float tensors for test inputs.
    @staticmethod
    def rand(size, device="cpu", dtype=torch.float):
        # [FUNCTION DOC] HELPER: Returns uniform random tensor of the given size, device and
        # dtype.
        return torch.rand(size, device=device, dtype=dtype)

    @staticmethod
    def randn(size, device="cpu", dtype=torch.float):
        # [FUNCTION DOC] HELPER: Returns standard-normal random tensor of the given size, device,
        # dtype.
        return torch.randn(size, device=device, dtype=dtype)

    @staticmethod
    def randp1(size, device="cpu", dtype=torch.float):
        # [FUNCTION DOC] HELPER: Returns uniform random values shifted by 1 (range one to two),
        # keeping inputs positive for log, div and similar ops.
        return torch.rand(size, device=device, dtype=dtype) + 1


# Tests vmap(op, in_dims, out_dims)(*inputs) by comparing the output to a
# (slow) sequential map+stack fallback.
#
# check_view: Test if the first returned output is a view of the first input
# check_propagates_grad: Test if the operation propagates gradients.
def _vmap_test(
    self,
    op,
    inputs,
    in_dims=0,
    out_dims=0,
    check_view=False,
    check_propagates_grad=True,
):
    # [FUNCTION DOC] HELPER: Compares vmap with the given in/out dims against reference_vmap using
    # assertEqual. If check_view, asserts each output's base is the base of the first input.
    # Unless check_propagates_grad is false, reruns with the first input cloned and requiring grad
    # and asserts the first output requires grad.
    result = vmap(op, in_dims, out_dims)(*inputs)
    reference_result = reference_vmap(op, inputs, in_dims, out_dims)
    self.assertEqual(result, reference_result)
    op_has_single_return = not isinstance(result, tuple)

    if check_view:
        result_as_tuple = (result,) if op_has_single_return else result
        for output in result_as_tuple:
            input0_base = inputs[0] if inputs[0]._base is None else inputs[0]._base
            self.assertTrue(
                output._base is input0_base,
                msg="result was not a view of the first input!",
            )

    if not check_propagates_grad:
        return
    # Assuming input[0] is a floating-point tensor. Check if the vmap
    # operation propagates the requires_grad flag to the zeroth output.
    # Some vmap operators are implemented in a way that assumes that
    # they are composite with respect to autograd. If the operator ever is
    # changed to not be composite with respect to autograd, then the
    # following check should fail.
    inputs_clone = list(inputs)
    inputs_clone[0] = inputs[0].clone().requires_grad_()
    result = vmap(op, in_dims, out_dims)(*inputs_clone)
    result_as_tuple = (result,) if op_has_single_return else result
    self.assertTrue(result[0].requires_grad)


def should_allow_vmap_fallback_usage(fn):
    # [FUNCTION DOC] HELPER: Returns the fn flag marking it as allowed to use the vmap fallback
    # (False when unset).
    return getattr(fn, "_allow_vmap_fallback_usage", False)


def allowVmapFallbackUsage(fn):
    # [FUNCTION DOC] HELPER: Decorator marking a test function as allowed to use the vmap
    # fallback, by setting a flag on it; returns the same function.
    fn._allow_vmap_fallback_usage = True
    return fn


# All tests of TestVmapBaseLegacy check that the slow vmap fallback is never invoked.
# This is so that we can incrementally add batching rules for operators to
# replace the slow vmap fallback path for said operators. To skip this check,
# please use the allowVmapFallbackUsage decorator.
#
# NB: Don't add tests to TestVmapBaseLegacy directly, unless you want them to run
# on every subclass of TestVmapBaseLegacy. Add them to e.g. TestVmapOperators.
#
# NB: TestVmapBaseLegacy is a nested class. This prevents test runners from picking
# it up and running it.
class Namespace:
    # [CLASS DOC] HELPER: Plain holder class whose only purpose is to nest the base test case so
    # test runners do not collect and run the base class itself. Has no state or methods.
    class TestVmapBaseLegacy(TestCase):
        # [CLASS DOC] Base test case (nested in Namespace so runners do not collect it directly)
        # whose test methods are wrapped, at construction, with a check that the vmap fallback
        # path is never used unless the test carries the allow flag. Subclasses add the actual
        # tests.
        def __init__(self, method_name="runTest"):
            # [FUNCTION DOC] HELPER: Initializes the TestCase then, if the named method exists and
            # is not flagged as allowing the fallback, replaces it on the instance with a wrapped
            # version that fails on fallback warnings.
            super().__init__(method_name)

            test_method = getattr(self, method_name, None)
            if test_method is None:
                return

            if not should_allow_vmap_fallback_usage(test_method):
                setattr(
                    self,
                    method_name,
                    self._wrap_method_with_vmap_fallback_check(test_method),
                )

        def _wrap_method_with_vmap_fallback_check(self, method):
            # [FUNCTION DOC] HELPER: Returns a bound wrapper of the given method that runs it with
            # all warnings recorded and fallback warnings enabled, then fails the test
            # (assertNotRegex) if any captured warning matches the performance-drop regex, using
            # an explanatory message.
            msg = (
                "Expected the test to not invoke the vmap fallback path, i.e., "
                "all of the operators being tested in this test should have batching "
                "rules implemented. If you are intentionally testing something to "
                "do with the fallback path, use allowVmapFallbackUsage. Otherwise, "
                "please make sure that batching rules are implemented for the "
                "operator(s) being tested."
            )

            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: The wrapper body; runs the wrapped method, then checks
                # the captured warnings.
                with warnings.catch_warnings(record=True) as wa:
                    warnings.simplefilter("always")
                    with EnableVmapFallbackWarnings():
                        method(*args, **kwargs)
                    for captured_warning in wa:
                        self.assertNotRegex(
                            str(captured_warning.message), FALLBACK_REGEX, msg
                        )

            return types.MethodType(wrapper, self)

        @allowVmapFallbackUsage
        def test_vmap_fallback_check_ok(self):
            # [FUNCTION DOC] Sanity check that a test explicitly allowed to use the fallback (via
            # var_mean) can run without being flagged.
            # One day we'll implement a batching rule for torch.var_mean.
            # When that happens, please change the example to use an
            # operator that doesn't have a batching rule implemented.
            op_using_fallback = torch.var_mean
            vmap(op_using_fallback)(torch.rand(3))

        def test_vmap_fallback_check(self):
            # [FUNCTION DOC] Checks the wrapping helper: a method that does nothing passes, while
            # one using the fallback fails with AssertionError.
            @self._wrap_method_with_vmap_fallback_check
            def no_fallback(self):
                # [FUNCTION DOC] HELPER: Empty method used to test the wrapper.
                pass

            # One day we'll implement a batching rule for torch.var_mean.
            # When that happens, please change the example to use an
            # operator that doesn't have a batching rule implemented.
            op_using_fallback = torch.var_mean

            @self._wrap_method_with_vmap_fallback_check
            def uses_fallback(self):
                # [FUNCTION DOC] HELPER: Method that vmaps the fallback-using var_mean op over a
                # random vector.
                vmap(op_using_fallback)(torch.rand(3))

            no_fallback(self)

            with self.assertRaises(AssertionError):
                uses_fallback(self)


class TestVmapOperatorsLegacy(Namespace.TestVmapBaseLegacy):
    # [CLASS DOC] Operator-level legacy vmap tests: each compares vmap results to the sequential
    # reference for many ops, covering various in_dims, out_dims, nested vmaps and view semantics.
    # Inherits the fallback check so these tests must not use the fallback.
    def _vmap_test(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards all arguments to the module-level vmap comparison helper
        # with self.
        return _vmap_test(self, *args, **kwargs)

    def _vmap_view_test(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Same as the vmap test helper, but also checks that outputs are
        # views of the first input.
        self._vmap_test(*args, **kwargs, check_view=True)

    def _test_unary(self, op, getter, device, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Runs the vmap comparison on a unary op for single and doubly
        # nested vmap with several in_dims / out_dims. Args: op, getter that builds input tensors,
        # device, and extra arguments forwarded to the vmap test helper.
        test = functools.partial(self._vmap_test, *args, **kwargs)
        B0, B1 = 7, 11

        # Single vmap, various in_dims / out_dims
        test(op, [getter([B0, 3], device)])
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2)
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2, out_dims=2)

        # Doubly nested vmap
        test(vmap(op), [getter([B0, B1], device)])
        test(vmap(op), [getter([B1, 2, 5, B0, 3], device)], in_dims=2)
        test(
            vmap(op, in_dims=2),
            [getter([2, 5, B0, B1, 3], device)],
            in_dims=2,
            out_dims=2,
        )

    def test_unary_pointwise_ops(self):
        # [FUNCTION DOC] Runs the unary helper over a list of pointwise ops (abs, trig, exp, log
        # families, rounding and so on), each with a getter producing inputs in a valid domain, on
        # CPU.
        cases = [
            (torch.abs, TensorFactory.randn),
            (torch.acos, TensorFactory.rand),
            (torch.asin, TensorFactory.rand),
            (torch.atan, TensorFactory.rand),
            (torch.ceil, TensorFactory.randn),
            (torch.cos, TensorFactory.rand),
            (torch.cosh, TensorFactory.rand),
            (torch.digamma, TensorFactory.rand),
            (torch.exp, TensorFactory.randn),
            (torch.expm1, TensorFactory.randn),
            (torch.floor, TensorFactory.randn),
            (torch.frac, TensorFactory.randn),
            (torch.lgamma, TensorFactory.rand),
            (torch.log, TensorFactory.randp1),
            (torch.log10, TensorFactory.randp1),
            (torch.log1p, TensorFactory.randp1),
            (torch.log2, TensorFactory.randp1),
            (torch.neg, TensorFactory.randn),
            (torch.reciprocal, TensorFactory.randp1),
            (torch.relu, TensorFactory.randn),
            (torch.round, TensorFactory.randn),
            (torch.rsqrt, TensorFactory.randp1),
            (torch.sigmoid, TensorFactory.randn),
            (torch.sign, TensorFactory.randn),
            (torch.sin, TensorFactory.rand),
            (torch.sinh, TensorFactory.rand),
            (torch.sqrt, TensorFactory.rand),
            (torch.tan, TensorFactory.rand),
            (torch.tanh, TensorFactory.rand),
            (torch.trunc, TensorFactory.randn),
        ]
        for op, getter in cases:
            self._test_unary(op, getter, "cpu")

    def test_clone(self):
        # [FUNCTION DOC] Checks clone vmap behavior for the default and the preserve and
        # contiguous memory formats, that contiguous-format results are contiguous per example,
        # and RuntimeError for channels-last formats.
        # Some basic tests
        self._test_unary(lambda x: x.clone(), TensorFactory.randn, "cpu")
        self._test_unary(
            lambda x: x.clone(memory_format=torch.preserve_format),
            TensorFactory.randn,
            "cpu",
        )
        self._test_unary(
            lambda x: x.clone(memory_format=torch.contiguous_format),
            TensorFactory.randn,
            "cpu",
        )

        # Test that the per-examples are contiguous when using torch.contiguous_format
        def clone_contiguous(x):
            # [FUNCTION DOC] HELPER: Clones x with contiguous memory format.
            return x.clone(memory_format=torch.contiguous_format)

        B0, B1 = 3, 5
        x = torch.randn(2, B0, 7)
        y = vmap(clone_contiguous, in_dims=1, out_dims=1)(x)
        self.assertTrue(y.movedim(1, 0).is_contiguous())
        self.assertTrue(y[:, 0, :].is_contiguous())

        x = torch.randn(2, B0, 7, B1)
        y = vmap(vmap(clone_contiguous, in_dims=2), in_dims=1)(x)
        self.assertTrue(y.is_contiguous())
        self.assertTrue(y[0][0].is_contiguous())

        msg = r"only supported with memory_format torch.preserve_format or torch.contiguous_format"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last))(torch.randn(B0))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last_3d))(
                torch.randn(B0)
            )

    def test_binary_pointwise_ops(self):
        # [FUNCTION DOC] Runs vmap comparisons for arithmetic binary ops (add, sub, mul, div, pow
        # and operator forms): mixed in_dims, nested, scalar-number overloads, and type promotion
        # cases.
        def get_number(getter):
            # [FUNCTION DOC] HELPER: Returns a Python number drawn from a scalar tensor made by
            # the getter.
            return getter([]).item()

        def make_case(op, input_getter=TensorFactory.randn):
            # [FUNCTION DOC] HELPER: Pairs an op with the input getter to use for it.
            return (op, input_getter)

        cases = [
            # Basic arithmetic
            make_case(torch.add),
            make_case(lambda x, y: x + y),
            make_case(torch.sub),
            make_case(lambda x, y: x - y),
            make_case(torch.mul),
            make_case(lambda x, y: x * y),
            make_case(torch.div, input_getter=TensorFactory.randp1),
            make_case(lambda x, y: x / y, input_getter=TensorFactory.randp1),
            make_case(torch.pow, input_getter=TensorFactory.randp1),
            make_case(lambda x, y: x**y, input_getter=TensorFactory.randp1),
        ]
        test = self._vmap_test

        for op, getter in cases:
            device = "cpu"
            B0, B1 = 7, 11

            # Single vmap: op(Tensor, Tensor)
            test(op, (getter([B0, 3], device), getter([B0, 3], device)))
            test(op, (getter([B0], device), getter([B0, 2, 3], device)))
            test(op, (getter([B0], device), getter([2, B0, 3], device)), in_dims=(0, 1))
            test(
                op,
                (getter([B0], device), getter([2, B0, 3], device)),
                in_dims=(0, 1),
                out_dims=1,
            )
            test(op, (getter([B0], device), getter([2, 3], device)), in_dims=(0, None))
            test(
                op, (getter([2, 3], device), getter([B0, 3], device)), in_dims=(0, None)
            )

            # Nested vmap: op(Tensor, Tensor)
            test(
                vmap(op), (getter([B0, B1, 2, 3], device), getter([B0, B1, 3], device))
            )
            test(
                vmap(op, in_dims=(None, 0)),
                (getter([B0, 2, 3], device), getter([B1, 3], device)),
                in_dims=(0, None),
            )

            # Python number overload: op(Tensor, Number) (and vice-versa)
            number = get_number(getter)
            self._test_unary(lambda t: op(t, number), getter, device)
            number = get_number(getter)
            self._test_unary(lambda t: op(number, t), getter, device)

            # Type promotion: op(Logical Scalar Tensor, Logical Scalar Tensor)
            test(op, (getter([B0], device), getter([B0], device, dtype=torch.double)))
            test(op, (getter([B0], device, dtype=torch.double), getter([B0], device)))
            test(op, (getter([B0], device), getter([B0], device)))

            # Type promotion: op(Tensor, Logical Scalar Tensor) (and vice-versa)
            test(op, (getter([B0, 2], device), getter([B0], device, torch.double)))
            test(op, (getter([B0], device, torch.double), getter([B0, 2], device)))

            if not torch.cuda.is_available():
                continue

            # TODO(rzou): fix the following
            # # Test cross-device scalars
            # number = get_number(getter)
            # self._test_unary(lambda t: op(t, number), getter, device='cuda')
            # self._test_unary(lambda t: op(number, t), getter, device='cuda')
            # self._test_unary(lambda t: op(t, torch.tensor(number)), getter, device='cuda')

    def test_as_strided(self):
        # [FUNCTION DOC] Checks the as_strided batching rule against equivalent view ops (expand,
        # transpose, select) for various strides and offsets, nested vmap, and RuntimeErrors for
        # malformed size/stride lengths, batch dims not at the front, and out-of-bounds access.
        def _test(sizes, strides, offset, tensor, lambd):
            # [FUNCTION DOC] HELPER: Asserts that vmapped as_strided shares its base with the
            # reference view function and gives equal values.
            result = vmap(lambda t: t.as_strided(sizes, strides, offset))(tensor)
            expected = vmap(lambd)(tensor)
            self.assertTrue(result._base is expected._base)
            self.assertEqual(result, expected)

        # single vmap test
        B0 = 5
        tensors = [
            # contiguous
            torch.randn(B0, 2, 3),
            # non-contiguous
            torch.randn(B0, 3, 2).transpose(1, 2),
            # non-zero storage offset
            torch.randn(2, B0, 2, 3)[1],
            # non-contiguous strides, zero storage offset
            torch.randn(B0, 2, 4, 3, 7)[:, :, 0, :, 0],
            # non-contiguous strides, non-zero storage offset
            torch.randn(B0, 2, 4, 3, 7)[:, :, 2, :, 1],
        ]

        for x in tensors:
            S0, S1 = x.stride()[1:]
            offset = x.storage_offset()

            # Broadcast
            _test(
                [5, 5, 2, 3], [0, 0, S0, S1], offset, x, lambda x: x.expand(5, 5, 2, 3)
            )
            # transpose
            _test([3, 2], [S1, S0], offset, x, lambda x: x.transpose(0, 1))
            # select
            _test([2], [S0], offset + S1, x, lambda x: x[:, 1])

        # Nested vmap test
        B1 = 7
        x = torch.randn(B1, B0, 2, 3)
        S0, S1 = x.stride()[2:]
        result = vmap(
            vmap(lambda t: t.as_strided([5, 5, 2, 3], [0, 0, S0, S1])), in_dims=1
        )(x)
        expected = vmap(vmap(lambda t: t.expand(5, 5, 2, 3)), in_dims=1)(x)
        self.assertTrue(result._base is expected._base)
        self.assertEqual(result, expected)

        # Check that mal-formatted size/strides doesn't crash
        with self.assertRaisesRegex(
            RuntimeError, "size and stride must have the same length"
        ):
            x = torch.randn(B0, 2, 3).transpose(0, 1)
            vmap(lambda x: x.as_strided([1, 1, 1], [1, 1]))(x)

        # Sanity check #1: we require the batch dims to be at the front of the
        # tensor (in memory layout).
        msg = "batch dims being vmapped over are at the front of the tensor"
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(2, B0, 3).transpose(0, 1)
            vmap(lambda x: x.as_strided([2, 3], [B0 * 3, 1]))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 2, 3, B1).movedim(3, 1)
            vmap(vmap(lambda x: x.as_strided([2, 3], [B1 * 3, B1])))(x)

        # All the Sanity check #2{a,b,c} cases check that
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # doesn't index memory that is out of bounds of xs[i]. This condition
        # is important to the correctness of the as_strided batching rule
        # (see NOTE: [When will the as_strided_batching_rule fail?])

        # Sanity check #2a: The maximum indexable location of
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is less than or equal to the maximum indexable location of xs[i].
        msg = "This is not supported inside of vmap"
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 3)
            vmap(lambda x: x.as_strided([3], [1], 1))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 3, 5)
            vmap(lambda x: x.as_strided([4, 4], [4, 1], 0))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, B1, 3, 5)
            vmap(vmap(lambda x: x.as_strided([4, 4], [4, 1], 0)))(x)

        # Sanity check #2b: The min indexable location of
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is greater than or equal to the min indexable location of xs[i].
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(2, B0, 3)[1]
            vmap(lambda x: x.as_strided([3], [1], B0 * 3 - 1))(x)

        # Sanity check #2c:
        # xs[i] is a zero-dim tensor, but
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is not
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 0, 3)
            vmap(lambda x: x.as_strided([3], [1]))(x)

    def test_bmm(self):
        # [FUNCTION DOC] Checks bmm: RuntimeError for shape mismatch, and correctness when the
        # left, right or both args are vmapped, including nested.
        op = torch.bmm
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = "Shape mismatch"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 3, 3, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(2, 5, 3)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(2, 5, 3)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5, 3), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5, 3), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(B0, 2, 5, 3)))
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(B0, B1, 2, 5, 3)),
            in_dims=(1, 0),
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 3, 5), torch.rand(B0, 2, 5, 3)),
            in_dims=(None, 0),
        )

    def test_cat(self):
        # [FUNCTION DOC] Checks cat under vmap with a variable number of mapped or unmapped
        # inputs, different dims (including negative), and nested vmap.
        test = self._vmap_test
        B0, B1 = 5, 7

        # Quick hack b/c vmap can't accept a list of tensors as an argument
        def get_op(dim):
            # [FUNCTION DOC] HELPER: Returns a function that concatenates its tensor arguments
            # along the dim.
            def op(*tensors):
                # [FUNCTION DOC] HELPER: Concatenates all given tensors along the enclosing dim.
                return torch.cat(tensors, dim=dim)

            return op

        test(get_op(0), (torch.rand(B0, 2), torch.rand(B0, 3)))
        test(get_op(0), (torch.rand(2), torch.rand(B0, 3)), in_dims=(None, 0))
        test(get_op(0), (torch.rand(2, 17), torch.rand(3, 17, B0)), in_dims=(None, 2))
        test(get_op(-1), (torch.rand(17, 2), torch.rand(17, 3, B0)), in_dims=(None, 2))
        test(
            vmap(get_op(0), in_dims=(0, None)),
            (torch.rand(B1, 2), torch.rand(B0, 3)),
            in_dims=(None, 0),
        )
        test(
            vmap(get_op(0), in_dims=(0, 0)),
            (torch.rand(B1, 2), torch.rand(B0, B1, 3)),
            in_dims=(None, 0),
        )

    def test_conj(self):
        # [FUNCTION DOC] Checks conj under vmap for float and complex inputs, and that real input
        # is returned with the same data pointer.
        op = torch.conj

        def run_test(dtype):
            # [FUNCTION DOC] HELPER: Runs single and nested vmap comparisons of conj for one
            # dtype.
            def get(shape):
                # [FUNCTION DOC] HELPER: Returns a random tensor of the shape in the enclosing
                # dtype.
                return torch.randn(shape, dtype=dtype)

            B0, B1 = 7, 11
            test = self._vmap_test

            # Single vmap, various in_dims / out_dims
            test(op, [get([B0, 3])])
            test(op, [get([2, 5, B0, 3])], in_dims=2)
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [get([B0, B1])])
            test(vmap(op), [get([B1, 2, 5, B0, 3])], in_dims=2)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)

        # correctness tests
        run_test(torch.float)
        run_test(torch.cfloat)

        # check that torch.conj on a non-complex tensor returns the same tensor
        real_tensor = torch.randn(3)
        result = vmap(op)(real_tensor)
        self.assertEqual(result.data_ptr(), real_tensor.data_ptr())

    def test_contiguous(self):
        # [FUNCTION DOC] Checks contiguous under vmap: unary comparisons, returns the original
        # tensor when per-example slices are already contiguous, and NYI RuntimeError for
        # channels-last memory formats.
        op = Tensor.contiguous

        self._test_unary(op, TensorFactory.randn, "cpu")

        # check that contiguous returns the original tensor if the per-examples
        # are already contiguous
        B0 = 3
        x = torch.randn(B0, 2, 5, 7)
        x = x.movedim(0, 2)
        result = vmap(Tensor.contiguous, in_dims=2, out_dims=2)(x)
        self.assertTrue(result is x)

        msg = "NYI: querying is_contiguous inside of vmap for memory_format"
        tensor = torch.randn(B0, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last_3d))(tensor)

    def test_stride(self):
        # [FUNCTION DOC] Checks that the per-example stride seen inside vmap excludes the batch
        # dim, for a contiguous and for a batch-dim-moved input.
        B0 = 3

        x = torch.randn(B0, 2, 5, 7)

        def foo(x):
            # [FUNCTION DOC] HELPER: Raises AssertionError unless the stride is the contiguous
            # 5-by-7 layout.
            if x.stride() != (7 * 5, 7, 1):
                raise AssertionError(f"stride mismatch: {x.stride()} != (35, 7, 1)")
            return x

        vmap(foo)(x)

        x = torch.randn(2, B0, 5, 7).movedim(1, 0)

        def bar(x):
            # [FUNCTION DOC] HELPER: Raises AssertionError unless the stride is the expected one
            # that accounts for the batch size in the outer stride.
            if x.stride() != (7 * 5 * B0, 7, 1):
                raise AssertionError(
                    f"stride mismatch: {x.stride()} != ({7 * 5 * B0}, 7, 1)"
                )
            return x

        vmap(bar)(x)

    def test_chunk(self):
        # [FUNCTION DOC] Checks chunk as a view op under vmap with several chunk sizes, dims and
        # nesting.
        test = self._vmap_view_test
        op = torch.chunk
        B0, B1, B2 = 7, 11, 13

        # tests for torch.split(self, split_size: int, dim)
        test(op, (torch.rand(B0, 2, 1024), 15, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 9, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 4, 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    def test_clamp(self):
        # [FUNCTION DOC] Runs the unary helper over clamp, clamp_min, clamp_max variants on random
        # input.
        clamp_cases = (
            (lambda t: t.clamp(min=-0.5), TensorFactory.randn),
            (lambda t: t.clamp(max=0.5), TensorFactory.randn),
            (lambda t: t.clamp(min=-0.5, max=0.5), TensorFactory.randn),
            (lambda t: t.clamp_min(min=-0.5), TensorFactory.randn),
            (lambda t: t.clamp_max(max=0.5), TensorFactory.randn),
        )
        for op, getter in clamp_cases:
            self._test_unary(op, getter, "cpu")

    def test_comparison_ops(self):
        # [FUNCTION DOC] Checks comparison ops (eq, gt, ge, le, lt, ne and operator forms) under
        # vmap with tensor and number operands, without checking grad propagation.
        test = functools.partial(self._vmap_test, check_propagates_grad=False)

        getter = TensorFactory.randn
        B0, B1 = 7, 11

        ops = (
            torch.eq,
            lambda x, y: x == y,
            torch.gt,
            lambda x, y: x > y,
            torch.ge,
            lambda x, y: x >= y,
            torch.le,
            lambda x, y: x <= y,
            torch.lt,
            lambda x, y: x < y,
            torch.ne,
            lambda x, y: x != y,
        )

        for op in ops:
            # Single vmap: op(Tensor, Tensor)
            test(op, (getter([B0, 3]), getter([B0, 3])))
            test(op, (getter([B0]), getter([B0, 2, 3])))
            test(op, (getter([B0]), getter([2, B0, 3])), in_dims=(0, 1))
            test(op, (getter([B0]), getter([2, B0, 3])), in_dims=(0, 1), out_dims=1)
            test(op, (getter([B0]), getter([2, 3])), in_dims=(0, None))
            test(op, (getter([2, 3]), getter([B0, 3])), in_dims=(0, None))

            # Nested vmap: op(Tensor, Tensor)
            test(vmap(op), (getter([B0, B1, 2, 3]), getter([B0, B1, 3])))
            test(
                vmap(op, in_dims=(None, 0)),
                (getter([B0, 2, 3]), getter([B1, 3])),
                in_dims=(0, None),
            )

            # test number as inputs
            number = getter([]).item()
            self._test_unary(
                lambda t: op(t, number), getter, "cpu", check_propagates_grad=False
            )

    def test_diagonal(self):
        # [FUNCTION DOC] Checks diagonal as a view op under vmap for several offsets, dims and
        # nesting.
        tensor = torch.randn(3, 5, 7, 11, 13)
        test = self._vmap_view_test
        op = torch.diagonal
        test(op, (tensor, 1, 0, 1), in_dims=(0, None, None, None))
        test(op, (tensor, 0, 2, -1), in_dims=(0, None, None, None))
        test(op, (tensor, 2, 1, 2), in_dims=(1, None, None, None))
        test(op, (tensor, 0, -2, -1), in_dims=(1, None, None, None), out_dims=1)
        test(vmap(lambda t: op(t, 0, 0, -1)), (tensor,), in_dims=1, out_dims=1)
        test(
            vmap(vmap(lambda t: op(t, 0, 0, 1), in_dims=1), in_dims=3),
            (tensor,),
            in_dims=1,
            out_dims=1,
        )

    def test_dot(self):
        # [FUNCTION DOC] Checks dot: RuntimeError for shape mismatch, and correctness with left,
        # right or both operands vmapped, including nested.
        op = torch.dot
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = "Shape mismatch"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 5), torch.rand(5)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 5), torch.rand(5)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(5), torch.rand(B1, B0, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 5), torch.rand(B0, 5)))
        test(vmap(op), (torch.rand(B1, B0, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 5), torch.rand(B0, 5)),
            in_dims=(None, 0),
        )

    def test_expand_as(self):
        # [FUNCTION DOC] Checks expand_as as a view op under vmap with mapped or unmapped operands
        # and nested vmaps.
        op = torch.Tensor.expand_as
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 1, 5), torch.rand(B0, 2, 3, 5)))
        test(op, (torch.rand(B0, 1, 5), torch.rand(2, 3, 5)), in_dims=(0, None))
        test(op, (torch.rand(1, 5), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(vmap(op), (torch.rand(B0, B1, 1, 5), torch.rand(B0, B1, 2, 3, 5)))
        test(
            vmap(op),
            (torch.rand(B0, B1, 1, 5), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(0, 1),
        )
        test(vmap(op), (torch.rand(B0, B1), torch.rand(B1, 2, 3, 5)), in_dims=(0, None))
        test(vmap(vmap(op)), (torch.rand(B0, B1, B2), torch.rand(B0, B1, B2, 2, 3, 5)))

    def test_fill_and_zero_inplace(self):
        # [FUNCTION DOC] Checks in-place fill_ and zero_ under vmap (single and nested), fill_
        # with a mapped value tensor, and RuntimeError when the written tensor is unmapped but the
        # value is mapped.
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B0, B1 = 7, 11
        ops = (
            lambda t: t.fill_(0.1),
            lambda t: t.fill_(torch.tensor(0.2)),
            lambda t: t.zero_(),
        )

        for op in ops:
            # Single vmap, various in_dims / out_dims
            test(op, [TensorFactory.randn([B0, 3])])
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2)
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [TensorFactory.randn([B0, B1])])
            test(vmap(op), [TensorFactory.randn([B1, 2, 5, B0, 3])], in_dims=2)
            test(
                vmap(op, in_dims=2),
                [TensorFactory.randn([2, 5, B0, B1, 3])],
                in_dims=2,
                out_dims=2,
            )

        # test when value is a batched tensor for fill_ operator
        B0, B1 = 3, 5
        test(Tensor.fill_, [TensorFactory.randn([B0, B1]), TensorFactory.randn(B0)])

        with self.assertRaisesRegex(
            RuntimeError, r"output with shape .+ doesn't match the broadcast shape"
        ):
            # Runtime Error is thrown when the tensor being written to isn't being vmapped over
            vmap(Tensor.fill_, (None, 0))(
                TensorFactory.randn([B0, B1]), TensorFactory.randn([B0])
            )

    def _test_complex_views(self, op, dtypes):
        # [FUNCTION DOC] HELPER: For each dtype, compares a complex view op (real, imag,
        # view_as_real) under single and nested vmap with several dims, including that outputs are
        # views.
        test = self._vmap_view_test

        def run_test(op, dtype):
            # [FUNCTION DOC] HELPER: Runs the view-comparison cases for the op with one dtype.
            def get(shape):
                # [FUNCTION DOC] HELPER: Returns a random tensor of the shape in the enclosing
                # dtype.
                return torch.randn(shape, dtype=dtype)

            B0, B1 = 7, 11

            # Single vmap, various in_dims / out_dims
            test(op, [get([B0, 3])])
            test(op, [get([3, B0])], in_dims=1)
            test(op, [get([2, 5, B0, 3])], in_dims=2)
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [get([B0, B1])])
            test(vmap(op), [get([B1, 2, 5, 3, B0])], in_dims=4)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)

        for dtype in dtypes:
            run_test(op, dtype)

    def test_real(self):
        # [FUNCTION DOC] Runs the complex-view helper on real for complex float and double.
        self._test_complex_views(torch.real, dtypes=[torch.cfloat, torch.cdouble])

    def test_imag(self):
        # [FUNCTION DOC] Runs the complex-view helper on imag for complex float and double.
        self._test_complex_views(torch.imag, dtypes=[torch.cfloat, torch.cdouble])

    def test_view_as_real(self):
        # [FUNCTION DOC] Runs the complex-view helper on view_as_real for complex float and
        # double.
        self._test_complex_views(
            torch.view_as_real, dtypes=[torch.cfloat, torch.cdouble]
        )

    def test_view_as_complex(self):
        # [FUNCTION DOC] Checks view_as_complex under vmap for float and double, including edge
        # cases of the batch dim next to or at the end of the tensor, and RuntimeError when the
        # size-2 dim lacks stride one.
        def run_test(dtype):
            # [FUNCTION DOC] HELPER: Runs the view_as_complex comparison cases for one dtype.
            def get(shape):
                # [FUNCTION DOC] HELPER: Returns a random tensor of the shape in the enclosing
                # dtype.
                return torch.randn(shape, dtype=dtype)

            op = torch.view_as_complex
            test = self._vmap_view_test
            B0, B1 = 7, 11

            # Single vmap, various in_dims / out_dims
            test(op, [get([B0, 3, 2])])
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2)
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [get([B0, B1, 2])])
            test(vmap(op), [get([B1, 2, 5, B0, 3, 2])], in_dims=2)
            test(
                vmap(op, in_dims=2), [get([2, 5, B0, B1, 3, 2])], in_dims=2, out_dims=2
            )

            # Interesting case #1: Batch dim directly before dim of size 2
            test(op, [get([3, B0, 2])], in_dims=1)
            test(vmap(op, in_dims=1), [get([3, B1, B0, 2])], in_dims=2)

            # Interesting case #2: Batch dim at end of tensor, success cases
            # view_as_complex requires that the dim with size 2 have stride 1
            # in order for the view to function properly
            test(op, [get([B0, 2]).transpose(0, 1)], in_dims=1)
            test(vmap(op, in_dims=1), [get([B0, B1, 2]).movedim(1, 2)])
            test(vmap(op, in_dims=2), [get([B0, 3, B1, 2]).movedim(2, 3)])

            # Interesting case #3: Batch dim at end of tensor, failure cases
            msg = "Tensor must have a last dimension with stride 1"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([2, B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op, in_dims=1), in_dims=1)(get([2, B0, B1]))

            # Invalid input: no dimension of size 2
            msg = "Input tensor must have one or more dimensions"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op)(get([B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op))(get([B0, B1]))

            # Invalid input: Batch dim has size 2, but the logical last dim does
            # not have size 2
            msg = "Tensor must have a last dimension of size 2"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([3, 2]))

        for dtype in [torch.float, torch.double]:
            run_test(dtype)

    def test_is_complex(self):
        # [FUNCTION DOC] Checks is_complex queried inside vmap returns a per-example constant:
        # ones for a complex input and zeros for a real one.
        ctensor = torch.randn(3, dtype=torch.cfloat)
        tensor = torch.randn(3)

        def foo(x):
            # [FUNCTION DOC] HELPER: Returns a scalar tensor one if x is complex, otherwise zero.
            if x.is_complex():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        self.assertEqual(vmap(foo)(ctensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(tensor), torch.tensor([0, 0, 0]))

    def test_is_floating_point(self):
        # [FUNCTION DOC] Checks is_floating_point inside vmap: float input gives ones, integer
        # input gives zeros.
        float_tensor = torch.tensor([1.0, 2.0, 3.0])
        long_tensor = torch.tensor([1, 2, 3])

        def foo(x):
            # [FUNCTION DOC] HELPER: Returns a scalar tensor one if x is floating point, otherwise
            # zero.
            if x.is_floating_point():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        self.assertEqual(vmap(foo)(float_tensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(long_tensor), torch.tensor([0, 0, 0]))

    def test_is_contiguous(self):
        # [FUNCTION DOC] Checks is_contiguous inside vmap reports per-example contiguity for
        # single and multiple batch dims (batch dim in front or elsewhere), that empty tensors are
        # contiguous, and that querying with a channels-last memory format raises a not-yet-
        # implemented RuntimeError.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns scalar one if x is contiguous else scalar zero.
            if x.is_contiguous():
                return torch.tensor(1.0)
            else:
                return torch.tensor(0.0)

        B0, B1 = 3, 5

        # Single batch dim
        contig = torch.randn(B0, 2, 7)
        self.assertEqual(vmap(foo)(contig), torch.ones(B0))

        noncontig = torch.randn(2, B0, 7)
        self.assertEqual(vmap(foo, in_dims=1)(noncontig), torch.zeros(B0))

        noncontig = torch.randn(2, B0, 7).movedim(1, 0)
        self.assertEqual(vmap(foo)(noncontig), torch.zeros(B0))

        noncontig = torch.randn(2, 7, B0)
        self.assertEqual(vmap(foo, in_dims=2)(noncontig), torch.zeros(B0))

        # Multiple batch dims
        contig = torch.randn(B0, B1, 3)
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))

        contig = torch.randn(B1, B0, 3)
        self.assertEqual(vmap(vmap(foo), in_dims=1)(contig), torch.ones(B0, B1))

        contig = torch.randn(B1, B0, 3).movedim(0, 1)
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))

        noncontig = torch.randn(B0, 3, B1)
        self.assertEqual(vmap(vmap(foo, in_dims=1))(noncontig), torch.zeros(B0, B1))

        # is_contiguous on empty tensor is True
        def bar(x):
            # [FUNCTION DOC] HELPER: Raises AssertionError if x is not contiguous; otherwise
            # returns x.
            if not x.is_contiguous():
                raise AssertionError("expected tensor to be contiguous")
            return x

        vmap(bar)(torch.randn(B0, 0, 3))
        vmap(bar, in_dims=1)(torch.randn(0, B0, 3))
        vmap(bar)(torch.randn(B0, 0, 3).mT)

        # is_contiguous with other memory formats
        def baz(x, memory_format):
            # [FUNCTION DOC] HELPER: Queries contiguity with the given memory format, discarding
            # the result, and returns x.
            x.is_contiguous(memory_format=memory_format)
            return x

        msg = "NYI: querying is_contiguous inside of vmap for memory_format"
        tensor = torch.randn(B0, 2, 7, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last_3d))(tensor)

    def test_movedim(self):
        # [FUNCTION DOC] Checks movedim as a view op under vmap for the integer and the list
        # overloads, with several in_dims and nested vmaps.
        op = torch.movedim
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        # movedim(tensor, int, int) variant
        test(op, (torch.rand(B0, 2, 5), 0, 1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 0, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5), 0, 1),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5, B2), 0, 1),
            in_dims=(2, None, None),
        )

        # movedim(tensor, intlist, intlist) variant
        test(op, (torch.rand(B0, 2, 3, 5), [1, 0], [0, 2]), in_dims=(0, None, None))
        test(op, (torch.rand(2, 3, B0, 5), [1, 0], [0, 2]), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5), [0, 1], [1, 0]),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5, B2), [0, 1], [1, 0]),
            in_dims=(2, None, None),
        )

    def test_mm(self):
        # [FUNCTION DOC] Checks mm: RuntimeError for shape mismatch, and correct results with the
        # left, right or both operands vmapped, including nested.
        op = torch.mm
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = "Shape mismatch"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(5, 2)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 5), torch.rand(5, 2)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5), torch.rand(B0, 5, 2)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5), torch.rand(B1, B0, 5, 2)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5, 2)))
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5, 2)),
            in_dims=(1, 0),
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 5), torch.rand(B0, 5, 2)),
            in_dims=(None, 0),
        )

    def test_mv(self):
        # [FUNCTION DOC] Checks mv (matrix times vector): shape mismatch RuntimeError, and
        # correctness with left, right or both operands vmapped, including nested.
        op = torch.mv
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = "Shape mismatch"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(5)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 5), torch.rand(5)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5), torch.rand(B1, B0, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5)))
        test(
            vmap(op), (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0)
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 5), torch.rand(B0, 5)),
            in_dims=(None, 0),
        )

    def test_narrow(self):
        # [FUNCTION DOC] Checks narrow as a view op under vmap with several dims, starts, lengths
        # and nesting.
        op = torch.narrow
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        test(op, (torch.rand(B0, 2, 5), -1, 1, 3), in_dims=(0, None, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1, 3), in_dims=(1, None, None, None))
        test(
            vmap(op, in_dims=(0, None, None, None)),
            (torch.rand(B1, 2, B0, 5), 1, 0, 0),
            in_dims=(2, None, None, None),
        )
        test(
            vmap(
                vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)
            ),
            (torch.rand(B1, 2, B0, 5, B2), -1, 2, 3),
            in_dims=(2, None, None, None),
        )

    def test_new_empty(self):
        # [FUNCTION DOC] Checks new_empty under vmap yields tensors of the expected shapes (values
        # are not compared since they are uninitialized), including empty size and nesting.
        # Empty is non-deterministic so we just check that the shape of the
        # output tensor is what we expect and that the vmap fallback isn't used.
        op = Tensor.new_empty

        B0, B1 = 7, 11

        result = vmap(lambda x: op(x, [2, 3]))(torch.randn(B0))
        self.assertEqual(result.shape, [B0, 2, 3])

        result = vmap(lambda x: op(x, []))(torch.randn(B0))
        self.assertEqual(result.shape, [B0])

        result = vmap(vmap(lambda x: op(x, [2, 3])))(torch.randn(B0, B1))
        self.assertEqual(result.shape, [B0, B1, 2, 3])

    def test_new_empty_strided(self):
        # [FUNCTION DOC] Checks new_empty_strided under vmap for contiguous, expanded and odd
        # size/stride combinations: result shape includes batch dims and the batch strides are
        # multiples of the per-example storage size.
        # Empty is non-deterministic so we just check that the size and shape
        # of the output are what we expect and that the vmap fallback isn't used
        B0, B1 = 7, 11

        def _test_single_vmap(size, stride, B0):
            # [FUNCTION DOC] HELPER: Checks shape and stride of single-vmap new_empty_strided
            # against the storage size of a plain strided tensor.
            x = torch.randn(B0)
            result = vmap(lambda x: x.new_empty_strided(size, stride))(x)
            S = torch.empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0] + size)
            self.assertEqual(result.stride(), [S] + stride)

        def _test_double_vmap(size, stride, B0, B1):
            # [FUNCTION DOC] HELPER: Same as the single check but for two nested vmaps, in default
            # and swapped in_dims order.
            x = torch.randn(B0, B1)
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)))(x)
            S = torch.empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0, B1] + size)
            self.assertEqual(result.stride(), [B1 * S, S] + stride)

            x = torch.randn(B1, B0)
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)), in_dims=1)(
                x
            )
            S = x.new_empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0, B1] + size)
            self.assertEqual(result.stride(), [B1 * S, S] + stride)

        # contiguous case
        _test_single_vmap([2, 3, 5], [3 * 5, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [3 * 5, 5, 1], B0, B1)

        # expanded
        _test_single_vmap([2, 3, 5], [0, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [0, 5, 1], B0, B1)

        # some of these cases are pretty strange, just verifying that if
        # empty_strided allows them then BatchedTensor.new_empty_strided
        # can as well
        for shape in [[2, 3, 4], [0, 2, 0]]:
            for strides in [[12, 4, 1], [2, 4, 6], [0, 0, 0]]:
                _test_single_vmap(shape, strides, B0)
                _test_double_vmap(shape, strides, B0, B1)

    def test_new_zeros(self):
        # [FUNCTION DOC] Checks new_zeros under vmap against the sequential reference for nonempty
        # and empty sizes and nesting, without a grad propagation check.
        op = Tensor.new_zeros
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B0, B1 = 7, 11

        test(lambda x: op(x, 2, 3), (torch.rand(B0),))
        test(lambda x: op(x, []), (torch.rand(B0),))
        test(vmap(lambda x: op(x, 3, 5)), (torch.rand(B0, B1),))

    def test_select(self):
        # [FUNCTION DOC] Checks select as a view op under vmap for various dims and nesting.
        op = torch.select
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5), 0, 0), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1), in_dims=(1, None, None))
        test(vmap(lambda t: op(t, 1, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(
            vmap(vmap(lambda t: op(t, 1, 1), in_dims=1)),
            (torch.rand(B1, 2, B0, B2, 5),),
            in_dims=2,
        )

    def test_stack(self):
        # [FUNCTION DOC] Checks stack under vmap with mapped or unmapped tensor arguments,
        # different dims (including negative) and nesting.
        test = self._vmap_test
        B0, B1 = 5, 7

        # Quick hack b/c vmap can't accept a list of tensors as an argument
        def get_op(dim):
            # [FUNCTION DOC] HELPER: Returns a function that stacks its tensor arguments along the
            # dim.
            def op(*tensors):
                # [FUNCTION DOC] HELPER: Stacks all given tensors along the enclosing dim.
                return torch.stack(tensors, dim=dim)

            return op

        test(get_op(0), (torch.rand(B0, 3), torch.rand(B0, 3)))
        test(get_op(0), (torch.rand(3), torch.rand(B0, 3)), in_dims=(None, 0))
        test(get_op(0), (torch.rand(2, 17), torch.rand(2, 17, B0)), in_dims=(None, 2))
        test(get_op(-1), (torch.rand(2, 17), torch.rand(2, 17, B0)), in_dims=(None, 2))
        test(
            vmap(get_op(0), in_dims=(0, None)),
            (torch.rand(B1, 2), torch.rand(B0, 2)),
            in_dims=(None, 0),
        )
        test(
            vmap(get_op(0), in_dims=(0, 0)),
            (torch.rand(B1, 2), torch.rand(B0, B1, 2)),
            in_dims=(None, 0),
        )

    def test_slice(self):
        # [FUNCTION DOC] Checks slicing as a view op under vmap in several dims with nesting.
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(lambda t: t[0:1], (torch.rand(B0, 3, 5),))
        test(lambda t: t[:, 1:3], (torch.rand(3, 5, B0),), in_dims=2)
        test(
            vmap(lambda t: t[:, 0:1], in_dims=2), (torch.rand(3, 5, B0, B1),), in_dims=2
        )
        test(
            vmap(vmap(lambda t: t[0:1], in_dims=2), in_dims=2),
            (torch.rand(3, 5, B0, B1, B2),),
            in_dims=2,
        )

    def test_squeeze(self):
        # [FUNCTION DOC] Checks squeeze as a view op under vmap including a batch size of one,
        # size-zero dims and nesting.
        test = self._vmap_view_test
        op = torch.squeeze
        B0, B1 = 1, 11
        test(op, (torch.rand(B0),))
        test(op, (torch.rand(B0, 3, 5),))
        test(op, (torch.rand(1, B0, 5),), in_dims=1)
        test(op, (torch.rand(B0, 0, 1, 5, 1),))
        test(op, (torch.rand(B0, 1, 1, 1, 1),))
        test(vmap(op), (torch.rand(B0, B1, 1),))
        test(vmap(op), (torch.rand(B1, 1, B0),), in_dims=2)

    def test_sum_dim(self):
        # [FUNCTION DOC] Checks sum over a dim (including empty dim tuple and negative dims) under
        # single and nested vmap with several in_dims/out_dims.
        test = self._vmap_test
        B0, B1 = 5, 7

        # Single vmap, various in_dims / out_dims
        test(lambda x: x.sum(()), [torch.randn([B0])])
        test(lambda x: x.sum(()), [torch.randn([B0, 2])])
        test(lambda x: x.sum(0), [torch.randn([B0])])
        test(lambda x: x.sum(-1), [torch.randn([B0])])
        test(lambda x: x.sum(0), [torch.randn([B0, 3])])
        test(lambda x: x.sum(-1), [torch.randn([2, 5, B0, 3])], in_dims=2)
        test(lambda x: x.sum(2), [torch.randn([2, 5, B0, 3])], in_dims=2, out_dims=2)

        # Doubly nested vmap
        test(vmap(lambda x: x.sum(())), [torch.randn([B0, B1])])
        test(vmap(lambda x: x.sum(0)), [torch.randn([B0, B1])])
        test(vmap(lambda x: x.sum(-1)), [torch.randn([B0, B1])])
        test(vmap(lambda x: x.sum(-2)), [torch.randn([B1, 2, 5, B0, 3])], in_dims=2)
        test(
            vmap(lambda x: x.sum(2), in_dims=2),
            [torch.randn([2, 5, B0, B1, 3])],
            in_dims=2,
            out_dims=2,
        )

    def test_reshape(self):
        # [FUNCTION DOC] Checks reshape under vmap; asserts the result is a view only in the cases
        # passing the view check, and nested usage.
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13
        op = torch.reshape
        test(op, (torch.rand(B0, 2 * 5), [2, 5]), in_dims=(0, None), check_view=True)
        test(
            op, (torch.rand(2, B0, 5), [1, 1, 10]), in_dims=(1, None), check_view=False
        )
        test(
            vmap(lambda t: t.reshape([-1])),
            (torch.rand(B0, B1, 2, 5),),
            check_view=True,
        )
        test(
            vmap(vmap(lambda t: t.reshape([-1]), in_dims=2), in_dims=1),
            (torch.rand(3, B1, 2, B2, 5, B0),),
            in_dims=5,
            check_view=False,
        )

    def test_reshape_as(self):
        # [FUNCTION DOC] Checks reshape_as under vmap with mapped and unmapped operands, view
        # checking only where a view is expected, and nested vmaps.
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13
        op = torch.Tensor.reshape_as
        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)), check_view=True)
        test(
            op,
            (torch.rand(2 * 5), torch.rand(B0, 2, 5)),
            in_dims=(None, 0),
            check_view=True,
        )
        test(
            op,
            (torch.rand(B0, 2 * 5), torch.rand(2, 5)),
            in_dims=(0, None),
            check_view=True,
        )

        test(
            op,
            (torch.rand(2, B0, 5), torch.rand(1, 1, 10)),
            in_dims=(1, None),
            check_view=False,
        )

        test(
            vmap(op),
            (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)),
            check_view=True,
        )
        test(
            vmap(vmap(op, in_dims=(2, None)), in_dims=(1, None)),
            (torch.rand(3, B1, 2, B2, 5, B0), torch.rand(B0, 3 * 2 * 5)),
            in_dims=(5, 0),
            check_view=False,
        )

    def test_result_type(self):
        # [FUNCTION DOC] Checks result_type under vmap against the sequential reference for
        # tensor/tensor, tensor/Python number, and tensor/scalar-tensor combinations of different
        # dtypes, wrapping the returned dtype into a scalar tensor so it can be compared.
        def scalar_tensor_with_dtype(op):
            # [FUNCTION DOC] HELPER: Wraps an op returning a dtype into a function returning a
            # scalar ones tensor of that dtype.
            def wrapped(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls the wrapped op with all arguments and returns a
                # scalar ones tensor with the resulting dtype.
                dtype = op(*args, **kwargs)
                return torch.ones([], dtype=dtype)

            return wrapped

        test = self._vmap_test
        op = scalar_tensor_with_dtype(torch.result_type)

        B0 = 2

        test(
            op,
            (torch.randn(B0), torch.randn(B0, dtype=torch.float64)),
            check_propagates_grad=False,
        )
        test(
            op,
            (torch.randn(B0), torch.randint(10, [B0], dtype=torch.int64)),
            check_propagates_grad=False,
        )

        test(lambda x: op(x, 1), (torch.randn(B0),), check_propagates_grad=False)
        test(lambda x: op(x, 1.6), (torch.randn(B0),), check_propagates_grad=False)

        test(
            lambda x: op(x, torch.tensor(1)),
            (torch.randn(B0),),
            check_propagates_grad=False,
        )
        test(
            lambda x: op(x, torch.tensor(1.6, dtype=torch.double)),
            (torch.randn(B0),),
            check_propagates_grad=False,
        )

        test(
            op,
            (torch.randn(B0, 2), torch.randn(B0, 2, dtype=torch.float64)),
            check_propagates_grad=False,
        )
        test(
            op,
            (torch.randn(B0, 2), torch.randint(10, [B0, 2], dtype=torch.int64)),
            check_propagates_grad=False,
        )

        test(lambda x: op(x, 1), (torch.randn(B0, 2),), check_propagates_grad=False)
        test(lambda x: op(x, 1.6), (torch.randn(B0, 2),), check_propagates_grad=False)

        test(
            lambda x: op(x, torch.tensor(1)),
            (torch.randn(B0, 2),),
            check_propagates_grad=False,
        )
        test(
            lambda x: op(x, torch.tensor(1.6, dtype=torch.double)),
            (torch.randn(B0, 2),),
            check_propagates_grad=False,
        )

        test(
            op,
            (torch.randn(B0, 2), torch.randn(B0, dtype=torch.float64)),
            check_propagates_grad=False,
        )
        test(
            op,
            (torch.randn(B0, 2), torch.randint(10, [B0], dtype=torch.int64)),
            check_propagates_grad=False,
        )

    @skipIfTorchDynamo("too slow")
    def test_tensor_split(self):
        # [FUNCTION DOC] Checks tensor_split as a view op under vmap, with an integer section
        # count and with a list of split indices, several dims and nesting. Skipped under
        # TorchDynamo as too slow.
        test = self._vmap_view_test
        op = torch.tensor_split
        B0, B1, B2 = 7, 11, 13

        # tests for torch.tensor_split(self, indices_or_sections: int, dim)
        test(op, (torch.rand(B0, 2, 1024), 5, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 150, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 256, 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

        # tests for torch.tensor_split(self, indices_or_sections: List[int], dim)
        test(
            op,
            (torch.rand(B0, 2, 1024), [50, 100, 378, 890], -1),
            in_dims=(0, None, None),
        )
        test(
            op,
            (torch.rand(2, B0, 1024), [50, 100, 212, 345, 0, 378, 890], 1),
            in_dims=(1, None, None),
        )
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), [50, 100, 212, 345, 0, 378, 890], 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, [4, 8, 9, 34, 29], 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    def test_split(self):
        # [FUNCTION DOC] Checks split as a view op under vmap with an integer split size and a
        # list of sizes, several dims and nesting.
        test = self._vmap_view_test
        op = torch.split
        B0, B1, B2 = 7, 11, 13

        # tests for torch.split(self, split_size: int, dim)
        test(op, (torch.rand(B0, 2, 1024), 101, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 130, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 256, 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

        # tests for torch.split(self, split_size: List[int], dim)
        test(op, (torch.rand(B0, 2, 1024), [1, 1020, 3], -1), in_dims=(0, None, None))
        test(
            op, (torch.rand(2, B0, 1024), [100] * 10 + [24], 1), in_dims=(1, None, None)
        )
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), [256] * 3 + [255], 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, [4] * 8 + [8] * 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    def test_trace(self):
        # [FUNCTION DOC] Checks trace under vmap with batch dims in different positions and
        # nesting.
        op = torch.trace
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13

        test(op, (torch.rand(B0, 2, 5),))
        test(op, (torch.rand(2, B0, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

    def test_transpose(self):
        # [FUNCTION DOC] Checks transpose as a view op under vmap for positive and negative dims,
        # nesting, and that a transpose on a per-example scalar returns the very same tensor.
        op = torch.transpose
        test = self._vmap_view_test

        B0, B1, B2 = 7, 11, 13
        test(lambda x: op(x, 0, 1), (torch.rand(B0, 2, 5),))
        test(lambda x: op(x, -1, -2), (torch.rand(B0, 2, 5),))
        test(lambda x: op(x, 3, 1), (torch.rand(B0, 2, 5, 4, 6),))
        test(lambda x: op(x, 1, 0), (torch.rand(2, B0, 5),), in_dims=1)
        test(vmap(lambda x: op(x, 0, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(
            vmap(vmap(lambda x: op(x, 0, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 5, B2),),
            in_dims=2,
        )

        # Special case: scalar tensor
        for dim1, dim2 in itertools.product([0, -1], [0, -1]):
            x = torch.rand(B0)
            result = vmap(lambda x: op(x, dim1, dim2))(x)
            self.assertTrue(result is x)

    def test_t(self):
        # [FUNCTION DOC] Checks the t op as a view under vmap with different batch positions and
        # nesting.
        op = torch.t
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5),))
        test(op, (torch.rand(2, B0, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 5, B2),), in_dims=2)

    def test_T_numpy(self):
        # [FUNCTION DOC] Checks the T attribute (reverse dim order) as a view under vmap with 2-D
        # and 3-D per-example tensors and nesting.
        def op(t):
            # [FUNCTION DOC] HELPER: Returns the T attribute of t.
            return t.T

        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 3, 5),))
        test(op, (torch.rand(2, B0, 3, 5),), in_dims=1)
        test(vmap(op), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(vmap(op), (torch.rand(B1, 2, B0, 3, 5),), in_dims=2)
        test(vmap(vmap(op, in_dims=2)), (torch.rand(B1, 2, B0, 3, B2, 5),), in_dims=2)

    def test_to(self):
        # [FUNCTION DOC] Checks the to method and casting methods (double, float, int, long)
        # under vmap, with device, dtype or other-tensor arguments; integer casts skip the grad
        # propagation check.
        test = self._vmap_test
        B0, B1 = 7, 11

        test(lambda t: t.to("cpu"), (torch.rand(B0),))
        test(lambda t: t.to(torch.double), (torch.rand(B0),))
        test(
            lambda t, o: t.to(o), (torch.rand(B0), torch.randn(B0, dtype=torch.float64))
        )
        test(
            lambda t, o: t.to(o),
            (torch.rand(B0), torch.randn(B0, dtype=torch.float64)),
            in_dims=(0, None),
        )
        test(vmap(lambda t: t.to(torch.double)), (torch.rand(B0, B1, 3),))

        # also test some casting methods
        test(lambda t: t.double(), (torch.rand(B0),))
        test(lambda t: t.float(), (torch.rand(B0),))
        test(lambda t: t.int(), (torch.rand(B0),), check_propagates_grad=False)
        test(lambda t: t.long(), (torch.rand(B0),), check_propagates_grad=False)

    def test_unfold(self):
        # [FUNCTION DOC] Checks unfold as a view op under vmap for a few dims, sizes, steps and
        # nesting.
        op = torch.Tensor.unfold
        test = self._vmap_view_test
        B0, B1, B2 = 3, 2, 5

        test(op, (torch.rand(B0, 7, 11), 0, 2, 1), in_dims=(0, None, None, None))
        test(op, (torch.rand(7, B0, 11), 1, 4, 2), in_dims=(1, None, None, None))
        test(
            vmap(op, in_dims=(0, None, None, None)),
            (torch.rand(B1, 7, B0, 11), 1, 5, 1),
            in_dims=(2, None, None, None),
        )
        test(
            vmap(
                vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)
            ),
            (torch.rand(B1, 7, B0, 11, B2), -1, 2, 4),
            in_dims=(2, None, None, None),
        )

    def test_unbind(self):
        # [FUNCTION DOC] Checks unbind as a view op under vmap, including a zero-size dim, various
        # dims and nesting.
        test = self._vmap_view_test
        op = torch.unbind
        B0, B1, B2 = 7, 11, 13

        test(op, (torch.rand(B0, 2, 1024), -1), in_dims=(0, None))
        test(op, (torch.rand(B0, 2, 0),))
        test(op, (torch.rand(2, B0, 7), 0), in_dims=(1, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 1023, B0, 5), 1),
            in_dims=(2, None),
        )
        test(
            vmap(vmap(lambda t: op(t, dim=1), in_dims=2)),
            (torch.rand(B1, 2, B0, 32, B2),),
            in_dims=2,
        )

    def test_view(self):
        # [FUNCTION DOC] Checks view under vmap: RuntimeError when the batched layout cannot
        # produce a correct view, and correct results with one-dim inference and nesting.
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        op = torch.Tensor.view

        # We should error out if the view would produce an incorrect result
        with self.assertRaises(RuntimeError):
            vmap(op, in_dims=(1, None))(torch.rand(2, B0, 5), [10])

        test(op, (torch.rand(B0, 2 * 5), [2, 5]), in_dims=(0, None))
        test(op, (torch.rand(B0, 4, 5), [1, 2, 1, 10]), in_dims=(0, None))
        test(vmap(lambda t: t.view([-1])), (torch.rand(B0, B1, 2, 5, 3),))
        test(
            vmap(vmap(lambda t: t.reshape([-1])), in_dims=1),
            (torch.rand(B2, B0, B1, 3, 2, 5),),
            in_dims=1,
        )

    def test_view_as(self):
        # [FUNCTION DOC] Checks view_as under vmap: RuntimeError when an incorrect view would
        # result, and correct results for mapped or unmapped operands and nesting.
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        op = torch.Tensor.view_as

        # We should error out if the view would produce an incorrect result
        with self.assertRaises(RuntimeError):
            vmap(op, in_dims=(1, 0))(torch.rand(2, B0, 5), torch.rand(B0, 10))

        test(op, (torch.rand(B0, 2 * 5), torch.rand(B0, 2, 5)))
        test(op, (torch.rand(2 * 5), torch.rand(B0, 2, 5)), in_dims=(None, 0))
        test(op, (torch.rand(B0, 2 * 5), torch.rand(2, 5)), in_dims=(0, None))

        test(op, (torch.rand(B0, 4, 5), torch.rand(2, 1, 1, 10)), in_dims=(0, None))

        test(vmap(op), (torch.rand(B0, B1, 2, 5), torch.randn(B0, B1, 10)))
        test(
            vmap(vmap(op, in_dims=(0, None)), in_dims=(0, None)),
            (torch.rand(B1, B2, B0, 3, 2, 5), torch.rand(B0, 3 * 2 * 5)),
            in_dims=(2, 0),
        )

    def test_no_random_op_support(self):
        # [FUNCTION DOC] Checks that vmap raises RuntimeError saying random operations are
        # unsupported for many random ops (out-of-place, in-place, on batched or captured tensors,
        # and factory functions).
        B0 = 2

        captured = torch.rand(3)

        random_ops = [
            # out-of-place on BatchedTensor
            (torch.bernoulli, (torch.rand(B0, 1),)),
            (lambda t: torch.bernoulli(t, p=0.5), (torch.rand(B0, 1),)),
            (lambda t: torch.multinomial(t, 2), (torch.rand(B0, 3),)),
            (torch.normal, (torch.randn(B0, 1), torch.randn(B0, 1))),
            (lambda t: torch.normal(t, 1.0), (torch.randn(B0, 1),)),
            (lambda t: torch.normal(0.0, t), (torch.randn(B0, 1),)),
            (torch.poisson, (torch.rand(B0, 1),)),
            (torch.rand_like, (torch.rand(B0, 1),)),
            (torch.randn_like, (torch.rand(B0, 1),)),
            (lambda t: torch.randint_like(t, 2), (torch.rand(B0, 1),)),
            (lambda t: torch.randint_like(t, 0, 2), (torch.rand(B0, 1),)),
            # out-of-place on captured tensor
            (lambda t: torch.bernoulli(captured), (torch.rand(B0),)),
            (lambda t: torch.bernoulli(captured, p=0.5), (torch.rand(B0),)),
            (lambda t: torch.multinomial(captured, 2), (torch.rand(B0),)),
            (lambda t: torch.normal(captured, captured), (torch.randn(B0),)),
            (lambda t: torch.normal(captured, 1.0), (torch.randn(B0),)),
            (lambda t: torch.normal(0.0, captured), (torch.randn(B0),)),
            (lambda t: torch.poisson(captured), (torch.rand(B0),)),
            (lambda t: torch.rand_like(captured), (torch.rand(B0),)),
            (lambda t: torch.randn_like(captured), (torch.rand(B0),)),
            (lambda t: torch.randint_like(captured, 2), (torch.rand(B0),)),
            (lambda t: torch.randint_like(captured, 0, 2), (torch.rand(B0),)),
            # in-place on BatchedTensor
            (lambda t: t.bernoulli_(), (torch.randn(B0, 1),)),
            (lambda t: t.cauchy_(), (torch.randn(B0, 1),)),
            (lambda t: t.exponential_(), (torch.randn(B0, 1),)),
            (lambda t: t.geometric_(0.5), (torch.randn(B0, 1),)),
            (lambda t: t.log_normal_(), (torch.randn(B0, 1),)),
            (lambda t: t.normal_(), (torch.randn(B0, 1),)),
            (lambda t: t.random_(), (torch.randn(B0, 1),)),
            (lambda t: t.random_(0, 2), (torch.randn(B0, 1),)),
            (lambda t: t.random_(2), (torch.randn(B0, 1),)),
            (lambda t: t.uniform_(), (torch.randn(B0, 1),)),
            # in-place on captured tensor
            (lambda t: captured.bernoulli_(), (torch.randn(B0),)),
            (lambda t: captured.cauchy_(), (torch.randn(B0),)),
            (lambda t: captured.exponential_(), (torch.randn(B0),)),
            (lambda t: captured.geometric_(0.5), (torch.randn(B0),)),
            (lambda t: captured.log_normal_(), (torch.randn(B0),)),
            (lambda t: captured.normal_(), (torch.randn(B0),)),
            (lambda t: captured.random_(), (torch.randn(B0),)),
            (lambda t: captured.random_(0, 2), (torch.randn(B0),)),
            (lambda t: captured.random_(2), (torch.randn(B0),)),
            (lambda t: captured.uniform_(), (torch.randn(B0),)),
            # factory functions
            (lambda t: torch.rand(1), (torch.randn(B0),)),
            (lambda t: torch.randn(1), (torch.randn(B0),)),
            (lambda t: torch.randint(5, [1]), (torch.randn(B0),)),
            (lambda t: torch.randperm(5), (torch.randn(B0),)),
        ]
        for op, args in random_ops:
            with self.assertRaisesRegex(
                RuntimeError, "vmap: We do not yet support calling random operations"
            ):
                vmap(op)(*args)


def construct_v(output, batch_size):
    # [FUNCTION DOC] HELPER: Returns a random normal tensor with the output's dtype and device,
    # with the leading batch dimension of the given size and the output's shape after it.
    return torch.randn(
        batch_size, *output.shape, dtype=output.dtype, device=output.device
    )


def as_tuple(x):
    # [FUNCTION DOC] HELPER: Returns x if a tuple, converts a list to a tuple, else wraps x in a
    # one-element tuple.
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def differentiable(args):
    # [FUNCTION DOC] HELPER: Returns a tuple of the entries of the (tuple-ified) args that are
    # Tensors requiring grad.
    return tuple(
        arg
        for arg in as_tuple(args)
        if isinstance(arg, torch.Tensor) and arg.requires_grad
    )


def _get_rand_no_zeros(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Returns a uniform random tensor for the given arguments, with values
    # raised to at least 0.1, honoring a requires_grad keyword (applied after clamping).
    requires_grad = kwargs.get("requires_grad", False)
    kwargs_without_requires_grad = kwargs.copy()
    kwargs_without_requires_grad["requires_grad"] = False
    result = torch.rand(*args, **kwargs_without_requires_grad)
    return result.clamp_min_(0.1).requires_grad_(requires_grad)


class TestVmapBatchedGradientLegacy(Namespace.TestVmapBaseLegacy):
    # [CLASS DOC] Tests of batched gradient computation (vmap over vector-Jacobian and vector-
    # Hessian products) for a range of ops, compared with the sequential reference. It is
    # instantiated per device type via the device-test machinery; tests take a device argument.
    def _vmap_test(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards to the module-level vmap comparison helper.
        return _vmap_test(self, *args, **kwargs)

    # Tests batched gradient computation of outputs = op(*args, **kwargs)
    # by comparing it to a sequential map+stack fallback.
    #
    # output_process_fn: a function that maps the outputs to the part
    #       that should be differentiated.
    # batch_size: the batch dim size for the batched grad
    def _batched_grad_test(
        self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3
    ):
        # [FUNCTION DOC] HELPER: Computes outputs of op on args (kwargs default to empty), keeps
        # the differentiable outputs after output_process_fn, builds random batched cotangents of
        # the given batch size, and checks vmapping a vector-Jacobian product function matches the
        # reference.
        if kwargs is None:
            kwargs = {}
        outputs = op(*args, **kwargs)
        outputs = differentiable(output_process_fn(outputs))
        batched_vectors = tuple(construct_v(out, batch_size) for out in outputs)

        def vector_jacobian_product(*vectors):
            # [FUNCTION DOC] HELPER: Returns gradients of the captured outputs w.r.t. the
            # differentiable args for the given cotangent vectors, retaining the graph.
            return torch.autograd.grad(
                outputs, differentiable(args), vectors, retain_graph=True
            )

        self._vmap_test(
            vector_jacobian_product, batched_vectors, check_propagates_grad=False
        )

    # Tests batched second grad computation of outputs = op(*args, **kwargs).
    # by comparing it to a sequential map+stack fallback.
    #
    # output_process_fn: a function that maps the outputs to the part
    #       that should be differentiated.
    # batch_size: the batch dim size for the batched grad
    #
    # NB: we only test computing batched gradients in the second gradient
    # computation. One specific use case that does this is computing the hessian
    # matrix of a scalar-valued function; this is useful in Bayesian Logistic
    # Regression.
    # It might be useful to have a test that computes batched first gradients and
    # then uses those to compute batched second gradients in the future.
    def _batched_grad_grad_test(
        self, op, args, kwargs=None, output_process_fn=lambda x: x, batch_size=3
    ):
        # [FUNCTION DOC] HELPER: Like the first-grad check but for the second derivative: takes
        # first grads with create_graph using ones, asserts at least one first grad exists, then
        # checks a vmapped vector-Hessian product matches the reference. Raises AssertionError if
        # no outputs result.
        if kwargs is None:
            kwargs = {}
        outputs = op(*args, **kwargs)
        outputs = differentiable(output_process_fn(outputs))
        ones = tuple(torch.ones_like(out) for out in outputs)
        # Same thing as summing together all of the outputs and calling .backward()
        first_grads = torch.autograd.grad(
            outputs, differentiable(args), ones, create_graph=True
        )
        first_grads = differentiable(first_grads)
        self.assertNotEqual(
            len(first_grads), 0, "None of the first grads depend on the input!"
        )

        batched_vectors = tuple(construct_v(grad, batch_size) for grad in first_grads)

        def vector_hessian_product(*vectors):
            # [FUNCTION DOC] HELPER: Differentiates the captured first grads w.r.t. the
            # differentiable args with the given vectors (unused allowed), drops None results,
            # raises AssertionError if none remain, and returns them.
            outputs = torch.autograd.grad(
                first_grads,
                differentiable(args),
                vectors,
                retain_graph=True,
                allow_unused=True,
            )
            outputs = tuple(out for out in outputs if out is not None)
            if len(outputs) == 0:
                raise AssertionError("outputs must not be empty")
            return outputs

        self._vmap_test(
            vector_hessian_product, batched_vectors, check_propagates_grad=False
        )

    def _test_arithmetic(self, op, device, test_grad_grad=True):
        # [FUNCTION DOC] HELPER: For a binary op, checks batched grads with tensor/tensor,
        # scalar/tensor and tensor/scalar operands; optionally also the batched second grad for
        # tensor/tensor.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        y = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        scalar = 3.14
        self._batched_grad_test(op, (x, y))
        self._batched_grad_test(op, (scalar, y))
        self._batched_grad_test(op, (x, scalar))

        if test_grad_grad:
            self._batched_grad_grad_test(op, (x, y))

    def test_add(self, device):
        # [FUNCTION DOC] Batched grad check for add (function and operator forms), without second
        # order.
        self._test_arithmetic(torch.add, device, test_grad_grad=False)
        self._test_arithmetic(lambda x, y: x + y, device, test_grad_grad=False)

    def test_sub(self, device):
        # [FUNCTION DOC] Batched grad check for sub (function and operator forms), without second
        # order.
        self._test_arithmetic(torch.sub, device, test_grad_grad=False)
        self._test_arithmetic(lambda x, y: x - y, device, test_grad_grad=False)

    def test_mul(self, device):
        # [FUNCTION DOC] Batched grad check for mul (function and operator forms), including
        # second order.
        self._test_arithmetic(torch.mul, device)
        self._test_arithmetic(lambda x, y: x * y, device)

    def test_div(self, device):
        # [FUNCTION DOC] Batched grad check for div (function and operator forms), including
        # second order.
        self._test_arithmetic(torch.div, device)
        self._test_arithmetic(lambda x, y: x / y, device)

    @allowVmapFallbackUsage
    def test_binary_cross_entropy(self, device):
        # [FUNCTION DOC] Batched first and second grad check for binary_cross_entropy on sigmoid
        # probabilities with a fixed target; allowed to use the vmap fallback.
        x = torch.sigmoid(torch.randn(3, 2, device=device, requires_grad=True))
        target = torch.rand(3, 2, device=device)

        op = functools.partial(F.binary_cross_entropy, target=target)

        self._batched_grad_test(op, (x,), {})
        self._batched_grad_grad_test(op, (x,), {})

    def test_expand(self, device):
        # [FUNCTION DOC] Batched grad check for expand of a 2x3 tensor to four dims.
        x = torch.randn(2, 3, device=device, requires_grad=True)

        def op(x):
            # [FUNCTION DOC] HELPER: Expands x to a larger shape.
            return x.expand(5, 5, 2, 3)

        self._batched_grad_test(op, (x,))

    @allowVmapFallbackUsage
    def test_index(self, device):
        # [FUNCTION DOC] Batched first and second grad check for advanced indexing of a squared
        # tensor with an index tensor; allowed to use the fallback.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        index = torch.tensor([[0, 0], [1, 1]], device=device)

        def op(x):
            # [FUNCTION DOC] HELPER: Squares x and indexes the result with the captured index
            # tensor.
            y = x * x
            return y[index]

        self._batched_grad_test(op, (x,))
        self._batched_grad_grad_test(op, (x,))

    def test_lgamma(self, device):
        # [FUNCTION DOC] Batched first and second grad check for lgamma.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(Tensor.lgamma, (x,))
        self._batched_grad_grad_test(Tensor.lgamma, (x,))

    def test_log(self, device):
        # [FUNCTION DOC] Batched first and second grad check for log, using inputs bounded away
        # from zero.
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(torch.log, (x,))
        self._batched_grad_grad_test(torch.log, (x,))

    def test_logsumexp(self, device):
        # [FUNCTION DOC] Batched first and second grad check for logsumexp over the last dim.
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)

        def op(x):
            # [FUNCTION DOC] HELPER: Applies logsumexp over the last dim.
            return torch.logsumexp(x, -1)

        self._batched_grad_test(op, (x,))
        self._batched_grad_grad_test(op, (x,))

    def test_log1p(self, device):
        # [FUNCTION DOC] Batched first and second grad check for log1p on inputs bounded away from
        # zero.
        x = _get_rand_no_zeros(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(torch.log1p, (x,))
        self._batched_grad_grad_test(torch.log1p, (x,))

    @allowVmapFallbackUsage
    def test_max(self, device):
        # [FUNCTION DOC] Batched first-order grad check for full max reduction; allowed to use the
        # fallback.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(torch.max, (x,))

    @allowVmapFallbackUsage
    def test_median(self, device):
        # [FUNCTION DOC] Batched first-order grad check for full median reduction; allowed to use
        # the fallback.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(torch.median, (x,))

    @allowVmapFallbackUsage
    def test_min(self, device):
        # [FUNCTION DOC] Batched first-order grad check for full min reduction; allowed to use the
        # fallback.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(torch.min, (x,))

    def test_permute(self, device):
        # [FUNCTION DOC] Batched grad check for permute of a 3-D tensor.
        x = torch.randn(2, 3, 5, requires_grad=True, device=device)

        def op(x):
            # [FUNCTION DOC] HELPER: Permutes dims of x.
            return x.permute(2, 0, 1)

        self._batched_grad_test(op, (x,))

    def test_reshape(self, device):
        # [FUNCTION DOC] Batched grad check for reshape of a 3-D tensor to 2-D.
        x = torch.randn(2, 3, 5, requires_grad=True, device=device)

        def op(x):
            # [FUNCTION DOC] HELPER: Reshapes x to 2-D.
            return x.reshape([2 * 3, 5])

        self._batched_grad_test(op, (x,))

    def test_sigmoid(self, device):
        # [FUNCTION DOC] Batched first and second grad check for sigmoid.
        x = torch.randn(2, 3, requires_grad=True, device=device)
        self._batched_grad_test(Tensor.sigmoid, (x,))
        self._batched_grad_grad_test(Tensor.sigmoid, (x,))

    def test_stack(self, device):
        # [FUNCTION DOC] Batched grad check for stack of two tensors.
        x = torch.randn(2, 3, device=device, requires_grad=True)
        y = torch.randn(2, 3, device=device, requires_grad=True)

        def op(x, y):
            # [FUNCTION DOC] HELPER: Stacks x and y into a new leading dim.
            return torch.stack([x, y])

        self._batched_grad_test(op, (x, y))

    def test_select(self, device):
        # [FUNCTION DOC] Batched grad check for select via indexing and select with positive and
        # negative dims.
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x[1], (x,))
        self._batched_grad_test(lambda x: x.select(1, 2), (x,))
        self._batched_grad_test(lambda x: x.select(-1, 0), (x,))

    def test_slice(self, device):
        # [FUNCTION DOC] Batched grad check for slices via indexing in different dims and with
        # ellipsis.
        x = torch.randn(2, 3, 5, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x[0:1], (x,))
        self._batched_grad_test(lambda x: x[:, 1:3], (x,))
        self._batched_grad_test(lambda x: x[..., 1:3], (x,))

    def test_trace(self, device):
        # [FUNCTION DOC] Batched grad check for trace.
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(Tensor.trace, (x,))

    def test_threshold(self, device):
        # [FUNCTION DOC] Batched grad check for the threshold function.
        x = torch.randn(2, 3, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: F.threshold(x, 0.5, 0.0), (x,))

    @allowVmapFallbackUsage
    def test_inplace_on_view(self, device):
        # [FUNCTION DOC] Batched first and second grad check for a function that applies cos in
        # place on a view of a product; allowed to use the fallback.
        leaf = torch.randn(4, 5, requires_grad=True)

        def func(leaf):
            # [FUNCTION DOC] HELPER: Squares the leaf, takes its first row as a view, applies cos
            # in place on the view and returns it.
            # Make sure the function is non-trivially twice differentiable
            base = leaf * leaf
            view = base[0]
            view.cos_()
            return view

        self._batched_grad_test(func, (leaf,), {})
        self._batched_grad_grad_test(func, (leaf,), {})

    @allowVmapFallbackUsage
    def test_inplace_manyview(self, device):
        # [FUNCTION DOC] Batched first and second grad check for in-place cos applied to a chain
        # of many views (transpose, select, diagonal, strided slice); allowed to use the fallback.
        leaf = torch.randn(4, 4, 5, requires_grad=True)

        def func(leaf):
            # [FUNCTION DOC] HELPER: Squares the leaf, takes a chain of views, applies cos in
            # place on the last view, returns it.
            # Make sure the function is non-trivially twice differentiable
            base = leaf * leaf
            view = base.transpose(0, 2)
            view = view[1]
            view = view.diagonal()
            view = view[::2]
            view.cos_()
            return view

        self._batched_grad_test(func, (leaf,), {})
        self._batched_grad_grad_test(func, (leaf,), {})

    def test_diagonal(self, device):
        # [FUNCTION DOC] Batched grad check for diagonal with different offsets and dims.
        x = torch.randn(4, 5, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x.diagonal(1, 0, 1), (x,))

        x = torch.randn(3, 4, 5, device=device, requires_grad=True)
        self._batched_grad_test(lambda x: x.diagonal(0, -1, -2), (x,))

    @allowVmapFallbackUsage
    def test_unrelated_output(self, device):
        # [FUNCTION DOC] Checks that vmapped autograd.grad of an output unrelated to the input
        # yields zeros of the right batch shape (allow_unused yields no gradient); allowed to use
        # the fallback.
        B0 = 3
        x = torch.randn([], requires_grad=True)
        y = torch.randn([], requires_grad=True)
        gy = torch.randn(B0, requires_grad=True)

        def vjp(v):
            # [FUNCTION DOC] HELPER: Returns the gradient of the captured y w.r.t. x for cotangent
            # v, or zeros if unused.
            (res,) = torch.autograd.grad(y, x, v, allow_unused=True)
            return torch.zeros_like(x) if res is None else res

        result = vmap(vjp)(gy)
        self.assertEqual(result, torch.zeros(B0, *x.shape, device=device))

    @allowVmapFallbackUsage
    def test_unrelated_output_multiple_grad(self, device):
        # [FUNCTION DOC] Same as the unrelated-output check but first calls the vjp function once
        # on an unbatched slice before vmapping, i.e. it takes multiple grads; allowed to use the
        # fallback.
        B0 = 3
        x = torch.randn([], requires_grad=True)
        y = torch.randn([], requires_grad=True)
        gy = torch.randn(B0, requires_grad=True)

        def vjp(v):
            # [FUNCTION DOC] HELPER: Returns the gradient of the captured y w.r.t. x for cotangent
            # v, or zeros if unused.
            (res,) = torch.autograd.grad(y, x, v, allow_unused=True)
            return torch.zeros_like(x) if res is None else res

        _ = vjp(gy[0])
        result = vmap(vjp)(gy)
        self.assertEqual(result, torch.zeros(B0, *x.shape, device=device))


instantiate_device_type_tests(TestVmapBatchedGradientLegacy, globals(), None)

if __name__ == "__main__":
    run_tests()
