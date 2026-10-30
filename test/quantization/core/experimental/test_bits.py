# Owner(s): ["oncall: quantization"]
# [MODULE DOC] Device-generic tests for the raw bits dtypes (bits1x8, bits2x4, bits4x2,
# bits8, bits16): creation, views, transpose/contiguous/clone, concatenation, and use as
# the backing dtype of a tensor subclass. Defines a bits16 tensor subclass helper and
# instantiates the device-type test classes at import time.

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_map

import itertools

class Int16Tensor(torch.Tensor):
    # [CLASS DOC] HELPER: Tensor subclass wrapping a bits16 tensor. Its dispatch hook views
    # all tensor arguments as int16, runs the operation without dispatch, and views tensor
    # results back as bits16, so ordinary arithmetic works on the opaque dtype.
    def __new__(cls, elem):
        # [FUNCTION DOC] HELPER: Creates the subclass instance from an element tensor sharing its
        # requires_grad flag. Raises AssertionError if the element dtype is not bits16.
        if elem.dtype != torch.bits16:
            raise AssertionError(f"Expected dtype torch.bits16, got {elem.dtype}")
        return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

    def __init__(self, elem):
        # [FUNCTION DOC] HELPER: Calls the base tensor initializer; ignores the element.
        super().__init__()

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Dispatch hook. Unwraps tensors in args and kwargs by viewing them
        # as int16 (under no-dispatch), calls the function, then wraps tensor outputs by viewing
        # them as bits16. Non-tensor values pass through unchanged. Returns the wrapped output.
        def unwrap(t):
            # [FUNCTION DOC] HELPER: Returns an int16 view of a tensor argument, other values as
            # is.
            if isinstance(t, torch.Tensor):
                with no_dispatch():
                    return t.view(torch.int16)
            return t
        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)

        with no_dispatch():
            out = func(*args, **kwargs)

        def wrap(t):
            # [FUNCTION DOC] HELPER: Returns a bits16 view of a tensor result, other values as is.
            if isinstance(t, torch.Tensor):
                with no_dispatch():
                    return t.view(torch.bits16)
            return t
        out = tree_map(wrap, out)
        return out

    # This most likely should be removed (and thus use the disabled impl)
    # but the test below fail under Dynamo in that case.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Delegates to the parent function-override handling unchanged; kept
        # because the tests reportedly fail under Dynamo without it.
        return super().__torch_function__(func, types, args, kwargs)

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a string with a demo prefix followed by the int16 view of
        # the tensor, computed without dispatch.
        with no_dispatch():
            self.view(torch.int16)
            return f"TensorSubclassDemo{self.view(torch.int16)}"


class TestBits(TestCase):
    # [CLASS DOC] Device-generic test case for the bits dtypes; instantiated per device by the
    # device type test machinery. Test methods take a device argument except the subclass test.
    def test_types(self, device):
        # [FUNCTION DOC] For each bits dtype, checks that zeros viewed as that dtype and empty
        # allocation work, and that a transposed view made contiguous, or cloned, has the same
        # underlying int8/int16 bit pattern as the transposed view. Uses the given device.
        bits_types = [torch.bits1x8, torch.bits2x4, torch.bits4x2, torch.bits8, torch.bits16]
        for bits_type in bits_types:
            _ = torch.zeros(20, dtype=torch.int32, device=device).view(bits_type)
            _ = torch.empty(20, dtype=bits_type, device=device)
            x = torch.randint(100, (20, 20), dtype=torch.int8, device=device).view(bits_type)
            y = x.t().contiguous()
            view_type = torch.int8 if x.element_size() == 1 else torch.int16
            self.assertEqual(x.t().view(view_type), y.view(view_type))
            y = x.t().clone()
            self.assertEqual(x.t().view(view_type), y.view(view_type))

    def test_cat(self, device):
        # [FUNCTION DOC] For each bits dtype, builds two random 512 by 512 tensors via integer
        # views, concatenates them along every dimension with and without transposing the second,
        # and checks the bit pattern equals concatenation done on the integer tensors.
        bits_types = [torch.bits1x8, torch.bits2x4, torch.bits4x2, torch.bits8, torch.bits16]
        for bits_type in bits_types:
            view_type = torch.int8 if bits_type.itemsize == 1 else torch.int16
            x_int = torch.randint(100, (512, 512), dtype=view_type, device=device)
            x = x_int.view(bits_type)
            y_int = torch.randint(100, (512, 512), dtype=view_type, device=device)
            y = y_int.view(bits_type)
            for dim, transpose in itertools.product(range(x_int.ndim), (True, False)):
                y_ref = y_int.t() if transpose else y_int
                y_b = y.t() if transpose else y
                z_ref = torch.cat([x_int, y_ref], dim=dim)
                z = torch.cat([x, y_b], dim=dim)
                self.assertEqual(z_ref, z.view(view_type))


    def test_subclass(self):
        # [FUNCTION DOC] Wraps a zero bits16 tensor in the subclass, adds then subtracts one, and
        # asserts the result is close to zeros of bits16 dtype. Takes no device argument.
        t = torch.zeros(20, dtype=torch.int16).view(torch.bits16)
        s = Int16Tensor(t)
        s = s + 1 - 1
        self.assertTrue(torch.allclose(s, torch.zeros(20, dtype=torch.bits16)))

instantiate_device_type_tests(TestBits, globals())


if __name__ == '__main__':
    run_tests()
