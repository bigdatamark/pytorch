# Owner(s): ["module: PrivateUse1"]
# [MODULE DOC] Tests for the Python-only PrivateUse1 backend support. At import time
# it registers the privateuseone device under the backend name npy, then implements a
# handful of aten ops for it via library impl registrations on top of NumPy arrays
# held by a tensor subclass. Tests exercise pinned-memory queries, a (skipped)
# autograd round trip and ldexp.
import unittest

import numpy as np

import torch
import torch._C
import torch._dynamo
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.backend_registration import _setup_privateuseone_for_python_backend


_setup_privateuseone_for_python_backend("npy")

aten = torch.ops.aten


# NOTE: From https://github.com/albanD/subclass_zoo/blob/main/new_device.py
# but using torch.library instead of `__torch_dispatch__`
class MyDeviceTensor(torch.Tensor):
    # [CLASS DOC] Tensor subclass acting as a device tensor for the privateuseone backend.
    # The real tensor is an empty wrapper created by the C++ accelerator helper; the NumPy
    # payload lives in the raw_data attribute, which op implementations unwrap and rewrap.
    @staticmethod
    def __new__(cls, size, dtype, raw_data=None, requires_grad=False):
        # [FUNCTION DOC] Static constructor: creates an empty wrapper tensor of the given size
        # and dtype via the C extension helper, rebinds its class to MyDeviceTensor and
        # returns it. The raw_data and requires_grad arguments are ignored here (handled in
        # the init method).
        # Use a meta Tensor here to be used as the wrapper
        res = torch._C._acc.create_empty_tensor(size, dtype)
        res.__class__ = MyDeviceTensor
        return res

    def __init__(self, size, dtype, raw_data=None, requires_grad=False):
        # [FUNCTION DOC] Initializer: stores the optional user-provided NumPy payload on the
        # raw_data attribute. The size, dtype and requires_grad arguments are accepted but
        # unused.
        # Store any provided user raw_data
        self.raw_data = raw_data

    def __repr__(self):
        # [FUNCTION DOC] Returns a string made of the class name followed by the string form
        # of the stored raw_data payload. The str form is aliased to this method.
        return "MyDeviceTensor" + str(self.raw_data)

    __str__ = __repr__


def wrap(arr, shape, dtype):
    # [FUNCTION DOC] HELPER: Wraps a NumPy array into a MyDeviceTensor with the given
    # shape and dtype. Despite the comment about float32, the dtype argument is passed
    # through unchanged.
    # hard code float32 for tests
    return MyDeviceTensor(shape, dtype, arr)


def unwrap(arr):
    # [FUNCTION DOC] HELPER: Returns the NumPy payload (raw_data attribute) of a
    # MyDeviceTensor.
    return arr.raw_data


# Add some ops
@torch.library.impl("aten::add.Tensor", "privateuseone")
def add(t1, t2):
    # [FUNCTION DOC] Implementation of add for two privateuseone tensors: adds the NumPy
    # payloads and returns a new wrapped tensor, always tagged as float32 regardless of
    # the inputs.
    out = unwrap(t1) + unwrap(t2)
    return wrap(out, out.shape, torch.float32)


@torch.library.impl("aten::mul.Tensor", "privateuseone")
def mul(t1, t2):
    # [FUNCTION DOC] Implementation of mul for two privateuseone tensors: multiplies the
    # NumPy payloads and returns a new wrapped float32 tensor.
    # If unsure what should be the result's properties, you can
    # use the super_fn (can be useful for type promotion)
    out = unwrap(t1) * unwrap(t2)
    return wrap(out, out.shape, torch.float32)


@torch.library.impl("aten::detach", "privateuseone")
def detach(self):
    # [FUNCTION DOC] Implementation of detach: wraps the same NumPy payload (no copy) in a
    # new float32 tensor.
    out = unwrap(self)
    return wrap(out, out.shape, torch.float32)


@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    # [FUNCTION DOC] Implementation of empty_strided: allocates an uninitialized NumPy
    # array of the requested size (the stride and other keyword arguments are ignored) and
    # wraps it with the requested dtype, defaulting to float32 when none is given.
    dtype = dtype if dtype is not None else torch.float32
    out = np.empty(size)
    return wrap(out, out.shape, dtype)


@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(a, b):
    # [FUNCTION DOC] Implementation of the copy-from op. When the source is an npy device
    # tensor its payload is used, otherwise the source is converted with its numpy method.
    # The payload is stored into the destination tensor's raw_data. Returns nothing.
    if a.device.type == "npy":
        npy_data = unwrap(a)
    else:
        npy_data = a.numpy()
    b.raw_data = npy_data


@torch.library.impl("aten::view", "privateuseone")
def _view(a, b):
    # [FUNCTION DOC] Implementation of view: wraps the source's payload (second argument
    # ignored) in a tensor with the source's shape and dtype. The payload is not reshaped.
    ans = unwrap(a)
    return wrap(ans, a.shape, a.dtype)


@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(
    size, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    # [FUNCTION DOC] Implementation of empty with memory format: allocates an
    # uninitialized NumPy array of the given size and wraps it with the requested dtype
    # (default float32). The other keyword arguments are ignored.
    dtype = dtype if dtype is not None else torch.float32
    ans = np.empty(size)
    return wrap(ans, ans.shape, dtype)


@torch.library.impl("aten::sum", "privateuseone")
def sum_int_list(*args, **kwargs):
    # [FUNCTION DOC] Implementation of sum: sums the whole payload of the first positional
    # argument (dim and other arguments ignored) and wraps the result as float32.
    ans = unwrap(args[0]).sum()
    return wrap(ans, ans.shape, torch.float32)


@torch.library.impl("aten::ones_like", "privateuseone")
def ones_like(
    self, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    # [FUNCTION DOC] Implementation of ones_like: builds a NumPy array of ones with the
    # shape of the input payload and wraps it as float32. Keyword arguments are ignored.
    ans = np.ones_like(unwrap(self))
    return wrap(ans, ans.shape, torch.float32)


@torch.library.impl("aten::expand", "privateuseone")
def expand(self, size, *, implicit=False):
    # [FUNCTION DOC] Implementation of expand: broadcasts the payload to the given size
    # and wraps the result as float32. The implicit flag is ignored.
    ans = np.broadcast_to(self.raw_data, size)
    return wrap(ans, ans.shape, torch.float32)


@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(self, size, stride, storage_offset=None):
    # [FUNCTION DOC] Implementation of as_strided: builds a strided NumPy view over the
    # payload using the given size and stride (storage_offset is ignored) and wraps it as
    # float32.
    ans = np.lib.stride_tricks.as_strided(self.raw_data, size, stride)
    return wrap(ans, ans.shape, torch.float32)


@torch.library.impl("aten::pow.Scalar", "privateuseone")
def pow_scalar(scalar, exponent):
    # [FUNCTION DOC] Implementation of pow with a scalar base: raises the scalar to the
    # power of the exponent tensor's payload and wraps the result as float32.
    ans = np.power(scalar, unwrap(exponent))
    return wrap(ans, ans.shape, torch.float32)


class PrivateUse1BackendTest(TestCase):
    # [CLASS DOC] Test case for the Python PrivateUse1 backend using the ops registered
    # above.
    @classmethod
    def setupClass(cls):
        # [FUNCTION DOC] HELPER: Placeholder class setup hook (note the non-standard name, so
        # unittest never calls it); does nothing.
        pass

    def test_accessing_is_pinned(self):
        # [FUNCTION DOC] Checks that querying is_pinned on a CPU tensor does not raise with
        # this backend registered. No assertion on the value.
        a_cpu = torch.randn((2, 2))
        # Assert this don't throw:
        _ = a_cpu.is_pinned()

    @unittest.skip(
        "Disabled due to CI failures; see "
        "https://github.com/pytorch/pytorch/issues/190232"
    )
    def test_backend_simple(self):
        # [FUNCTION DOC] Moves two random CPU tensors to privateuseone, marks them as
        # requiring grad, adds and sums them, runs backward and checks the gradient payload
        # equals ones.
        # KNOWN ISSUE: skipped unconditionally (CI failures, tracked in a linked upstream
        #   issue).
        a_cpu = torch.randn((2, 2))
        b_cpu = torch.randn((2, 2))
        # Assert this don't throw:
        a = a_cpu.to("privateuseone")
        b = b_cpu.to("privateuseone")

        a.requires_grad = True
        b.requires_grad = True
        c = (a + b).sum()
        c.backward()
        self.assertTrue(np.allclose(a.grad.raw_data, np.ones((2, 2))))

    def test_ldexp(self):
        # [FUNCTION DOC] Moves a float32 tensor and an int32 tensor to privateuseone and calls
        # ldexp, then checks the result payload is approximately the values 2 and 8. The
        # assertion is run with Dynamo disabled to avoid fake tensor crashes.
        a_cpu = torch.tensor([1.0, 2.0], dtype=torch.float32)
        b_cpu = torch.tensor([1, 2], dtype=torch.int32)
        a = a_cpu.to("privateuseone")
        b = b_cpu.to("privateuseone")
        res = torch.ldexp(a, b)
        # Disable Dynamo tracing on test assertions to prevent compile-time FakeTensor crashes
        torch._dynamo.disable(self.assertTrue)(
            np.allclose(unwrap(res), np.array([2.0, 8.0]))
        )


if __name__ == "__main__":
    run_tests()
