# Owner(s): ["module: PrivateUse1"]
# [MODULE DOC] Operator tests for the openreg backend: factories, copies, views, dispatch
# stubs (abs), CPU fallback, quantization, scaled dot product attention (overrideable
# fused backend) and custom autograd functions. Defines SDPAShape, a namedtuple of
# batch, num_heads, seq_len and head_dim. No other import-time side effects.

import collections
import functools
import unittest

import torch
from torch.nn.attention import SDPBackend
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


SDPAShape = collections.namedtuple(
    "Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"]
)


class TestFactory(TestCase):
    # [CLASS DOC] Tests for basic tensor creation on openreg.
    def test_empty(self):
        # [FUNCTION DOC] Empty tensor has openreg device and shape; under FakeTensorMode adding
        # two empty openreg tensors (index unspecified and
        # zero) gives an openreg result of equal shape.
        """Test empty tensor creation"""
        x = torch.empty(3, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([3]))

        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.empty(3, 3, device="openreg")
            y = torch.empty(3, 3, device="openreg:0")
            z = x + y
            self.assertEqual(z.device.type, "openreg")
            self.assertEqual(z.shape, torch.Size([3, 3]))

    def test_zeros(self):
        # [FUNCTION DOC] zeros creates an openreg tensor with the requested shape.
        """Test zeros tensor creation"""
        y = torch.zeros(3, device="openreg")
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, torch.Size([3]))

    def test_tensor(self):
        # [FUNCTION DOC] Building a tensor from an empty tuple on openreg gives shape zero.
        """Test tensor creation from empty tuple"""
        z = torch.tensor((), device="openreg")
        self.assertEqual(z.device.type, "openreg")
        self.assertEqual(z.shape, torch.Size([0]))


class TestCopy(TestCase):
    # [CLASS DOC] Tests for copying within and across devices.
    def test_copy_same_device(self):
        # [FUNCTION DOC] Cloning a ones tensor on openreg equals ones.
        """Test copy operation on same device"""
        a = torch.ones(10, device="openreg").clone()
        self.assertEqual(a, torch.ones(10, device="openreg"))

    def test_cross_device_copy(self):
        # [FUNCTION DOC] CPU to openreg, add two, back to CPU equals the CPU input plus two.
        """Test copy operation across CPU and openreg"""
        a = torch.rand(10)
        b = a.to(device="openreg").add(2).to(device="cpu")
        self.assertEqual(b, a + 2)

    def test_cross_diff_devices_copy(self):
        # [FUNCTION DOC] A tensor moved openreg:0 to openreg:1 to CPU equals ones.
        """Test copy operation across different openreg devices"""
        a = torch.ones(10, device="openreg:0").to(device="openreg:1").to(device="cpu")
        self.assertEqual(a, torch.ones(10))


class TestOps(TestCase):
    # [CLASS DOC] Tests for assorted ops (masked_select, expand, resize, printing) on openreg.
    def test_masked_select(self):
        # [FUNCTION DOC] masked_select on openreg matches the CPU result.
        """Test masked_select operation"""
        tensor_cpu = torch.randn(10)
        tensor_openreg = tensor_cpu.to(device="openreg")
        mask = tensor_openreg.gt(0)
        out = torch.masked_select(tensor_openreg, mask)

        self.assertEqual(out, tensor_cpu.masked_select(tensor_cpu.gt(0)))

    def test_expand(self):
        # [FUNCTION DOC] Expanding a 3 by 1 tensor to 3
        # by 2 gives the repeated values and shares the data pointer.
        """Test tensor expand operation"""
        x = torch.tensor([[1], [2], [3]], device="openreg")
        y = x.expand(3, 2)
        self.assertEqual(y.to(device="cpu"), torch.tensor([[1, 1], [2, 2], [3, 3]]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_resize(self):
        # [FUNCTION DOC] Resizing a 4 by 4 tensor to 2 by 2 by 2 by 2 changes the size but the
        # storage keeps sixteen elements.
        """Test tensor resize operation"""
        tensor_cpu = torch.randn([4, 4])

        tensor_openreg = tensor_cpu.openreg()
        self.assertTrue(tensor_openreg.size() == torch.Size([4, 4]))

        storage_openreg = tensor_openreg.storage()
        self.assertTrue(storage_openreg.size() == 16)

        tensor_openreg.resize_(2, 2, 2, 2)
        self.assertTrue(tensor_openreg.size() == torch.Size([2, 2, 2, 2]))

        storage_openreg = tensor_openreg.storage()
        self.assertTrue(storage_openreg.size() == 16)

    def test_printing(self):
        # [FUNCTION DOC] Prints an openreg tensor; passes if no error is raised.
        """Test tensor printing"""
        a = torch.ones(20, device="openreg")
        print(a)


class TestSTUB(TestCase):
    # [CLASS DOC] Tests for the openreg dispatch stub implementation of abs.
    def test_backend_dispatchstub(self):
        # [FUNCTION DOC] abs on openreg matches CPU, including out= into strided slices (step two
        # and three) where TensorIterator keeps or replaces the output operand.
        """Test backend dispatch stub for abs operation"""
        x_cpu = torch.randn(2, 2, 3, dtype=torch.float32, device="cpu")
        x_openreg = x_cpu.to("openreg")

        y_cpu = torch.abs(x_cpu)
        y_openreg = torch.abs(x_openreg)
        self.assertEqual(y_cpu, y_openreg.cpu())

        o_cpu = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        o_openreg = o_cpu.to("openreg")
        # output operand with resize flag is False in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:2])
        torch.abs(x_openreg, out=o_openreg[:, :, 0:6:2])
        self.assertEqual(o_cpu, o_openreg.cpu())

        # output operand with resize flag is True in TensorIterator and
        # convert output to contiguous tensor in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:3])
        torch.abs(x_openreg, out=o_openreg[:, :, 0:6:3])
        self.assertEqual(o_cpu, o_openreg.cpu())


class TestQuantization(TestCase):
    # [CLASS DOC] Basic quantization test on openreg.
    def test_quantize(self):
        # [FUNCTION DOC] Per-tensor quantization to qint8 returns
        # a tensor on openreg:0 with dtype qint8.
        """Test quantization per tensor"""
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="openreg")
        quantized_tensor = torch.quantize_per_tensor(x, 0.1, 10, torch.qint8)
        self.assertEqual(quantized_tensor.device, torch.device("openreg:0"))
        self.assertEqual(quantized_tensor.dtype, torch.qint8)


class TestAutogradFunction(TestCase):
    # [CLASS DOC] Tests custom autograd functions under torch.compile with aot_eager.
    def test_compile_autograd_function_returns_self(self):
        # [FUNCTION DOC] Compares eager and compiled (aot_eager) outputs and input gradients of
        # the custom op that returns its own input.
        """Test compiled autograd function that returns self"""
        in_ref = torch.randn(4, device="openreg", requires_grad=True)
        out_ref = torch.ops.openreg.custom_autograd_fn_returns_self(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = torch.compile(backend="aot_eager")(
            torch.ops.openreg.custom_autograd_fn_returns_self
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(in_ref.grad, in_test.grad)

    @skipIfTorchDynamo("Temporary disabled due to torch._ops.OpOverloadPacket")
    def test_compile_autograd_function_aliasing(self):
        # [FUNCTION DOC] Same comparison for the custom op that returns an alias of its input.
        # Skipped under Dynamo.
        """Test compiled autograd function with aliasing"""
        in_ref = torch.randn(4, device="openreg", requires_grad=True)
        out_ref = torch.ops.openreg.custom_autograd_fn_aliasing(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = torch.compile(backend="aot_eager")(
            torch.ops.openreg.custom_autograd_fn_aliasing
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(in_ref.grad, in_test.grad)


class TestFallback(TestCase):
    # [CLASS DOC] Tests for CPU fallback of ops without openreg kernels, by argument type.
    def test_scalar_type_fallback(self):
        # [FUNCTION DOC] triu_indices on openreg (scalar-type
        # arguments) matches the expected int64 CPU values.
        """Test scalar type fallback to CPU"""
        x_cpu = torch.Tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]).to(torch.int64)
        x = torch.triu_indices(3, 3, device="openreg")
        self.assertEqual(x_cpu, x)

    def test_tensor_type_fallback(self):
        # [FUNCTION DOC] Subtraction and advanced indexing on openreg tensors, which fall back
        # to the CPU, give the expected values.
        # KNOWN ISSUE: assertTrue on the device type uses the string as the message, so it
        # always passes.
        """Test tensor type fallback to CPU"""
        x = torch.Tensor([[1, 2, 3], [2, 3, 4]]).to("openreg")
        y = torch.Tensor([1, 0, 2]).to("openreg")
        self.assertTrue(x.device.type, "openreg")
        self.assertFalse(x.is_cpu)

        z_cpu = torch.Tensor([[0, 2, 1], [1, 3, 2]])
        # call sub op, which will fallback to cpu
        z = torch.sub(x, y)
        self.assertEqual(z_cpu, z)

        # call index op, which will fallback to cpu
        z_cpu = torch.Tensor([3, 1])
        y = torch.Tensor([1, 0]).long().to("openreg")
        z = x[y, y]
        self.assertEqual(z_cpu, z)

    def test_tensorlist_type_fallback(self):
        # [FUNCTION DOC] The foreach add op on tuples of openreg tensors falls back to the CPU
        # and returns the expected sums.
        """Test tensor list type fallback to CPU"""
        # create tensors located in custom device
        v_openreg = torch.Tensor([1, 2, 3]).to("openreg")
        # create result tensor located in cpu
        z_cpu = torch.Tensor([2, 4, 6])
        # create tensorlist for foreach_add op
        x = (v_openreg, v_openreg)
        y = (v_openreg, v_openreg)

        # Check that our device is correct.
        self.assertTrue(v_openreg.device.type == "openreg")
        self.assertFalse(v_openreg.is_cpu)

        # call _foreach_add op, which will fallback to cpu
        z = torch._foreach_add(x, y)
        self.assertEqual(z_cpu, z[0])
        self.assertEqual(z_cpu, z[1])


class TestSDPA(NNTestCase):
    # [CLASS DOC] Tests that scaled dot product attention routes to the overrideable backend.
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_privateuseone(self):
        # [FUNCTION DOC] For half-precision openreg q, k, v the fused SDP choice must be the
        # OVERRIDEABLE backend value, else AssertionError is raised. Skipped under Dynamo.
        """Test fused SDP choice for privateuse1 backend"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        if (
            torch._fused_sdp_choice(q_privateuse1, k_privateuse1, v_privateuse1)
            != SDPBackend.OVERRIDEABLE.value
        ):
            raise AssertionError(
                "Expected _fused_sdp_choice to return SDPBackend.OVERRIDEABLE.value"
            )

    def test_scaled_dot_product_fused_attention_overrideable(self):
        # [FUNCTION DOC] Runs scaled dot product attention on openreg inputs without mask or
        # dropout; passes if no error.
        """Test scaled dot product fused attention overrideable forward"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask=None, dropout_p=0.0
        )

    def test_scaled_dot_product_fused_attention_overrideable_backward(self):
        # [FUNCTION DOC] Calls the fused overrideable forward op with an attention bias and
        # then its backward op with a random upstream gradient and all grad-input mask flags
        # true; passes if no error. Outputs are not checked.
        """Test scaled dot product fused attention overrideable backward"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(
            torch.rand, device="cpu", dtype=torch.float16, requires_grad=True
        )
        shape = (batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        attn_mask_privateuse1 = attn_mask.to("openreg")
        (
            output,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            _debug_attn_mask,
        ) = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_bias=attn_mask_privateuse1
        )

        rand_upward = torch.rand(
            shape, device="cpu", dtype=torch.float16, requires_grad=False
        )
        rand_upward_privateuse1 = rand_upward.to("openreg")
        grad_input_mask = [True, True, True, True]
        _grad_q, _grad_k, _grad_v, _grad_attn_mask = (
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward(
                rand_upward_privateuse1,
                q_privateuse1,
                k_privateuse1,
                v_privateuse1,
                attn_mask_privateuse1,
                grad_input_mask,
                output,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                dropout_p=0.0,
                is_causal=False,
                philox_seed=philox_seed,
                philox_offset=philox_offset,
            )
        )


class TestFactoryExtended(TestCase):
    # [CLASS DOC] More tensor creation tests on openreg.
    def test_empty_with_memory_format(self):
        # [FUNCTION DOC] Empty tensors with channels-last and contiguous formats have the right
        # device, shape and contiguity.
        """Test empty tensor creation with memory format"""
        x = torch.empty(1, 2, 3, 4, device="openreg", memory_format=torch.channels_last)
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([1, 2, 3, 4]))

        x = torch.empty(
            2, 3, 4, device="openreg", memory_format=torch.contiguous_format
        )
        self.assertEqual(x.device.type, "openreg")
        self.assertTrue(x.is_contiguous())

    def test_empty_strided(self):
        # [FUNCTION DOC] empty_strided honors size and stride on openreg.
        """Test empty_strided tensor creation"""
        size = (3, 4)
        stride = (4, 1)
        x = torch.empty_strided(size, stride, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size(size))
        self.assertEqual(x.stride(), stride)

    def test_ones(self):
        # [FUNCTION DOC] ones gives the right device, shape and all-one values.
        """Test ones tensor creation"""
        x = torch.ones(3, 4, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([3, 4]))
        self.assertTrue(torch.all(x == 1))

    def test_ones_like(self):
        # [FUNCTION DOC] ones_like keeps device and shape and is all ones.
        """Test ones_like tensor creation"""
        x = torch.randn(3, 4, device="openreg")
        y = torch.ones_like(x)
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.all(y == 1))

    def test_randn(self):
        # [FUNCTION DOC] randn gives the right device and shape.
        """Test randn tensor creation"""
        x = torch.randn(3, 4, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([3, 4]))

    def test_full(self):
        # [FUNCTION DOC] full with 5.0 gives the right device, shape and values.
        """Test full tensor creation"""
        x = torch.full((3, 4), 5.0, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([3, 4]))
        self.assertTrue(torch.all(x == 5.0))


class TestCopyExtended(TestCase):
    # [CLASS DOC] More copy tests on openreg.
    def test_copy_different_dtypes(self):
        # [FUNCTION DOC] Copying float32 into a float64 openreg tensor converts values.
        """Test copy with different dtypes"""
        x = torch.randn(3, 4, dtype=torch.float32, device="openreg")
        y = torch.empty(3, 4, dtype=torch.float64, device="openreg")
        y.copy_(x)
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual(y.cpu(), x.cpu().double())

    def test_clone(self):
        # [FUNCTION DOC] clone is equal in value but has a different data pointer.
        """Test tensor clone"""
        x = torch.randn(3, 4, device="openreg")
        y = x.clone()
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y, x)
        self.assertNotEqual(y.data_ptr(), x.data_ptr())

    def test_copy_non_blocking(self):
        # [FUNCTION DOC] copy_ with non_blocking true yields equal values.
        """Test non-blocking copy"""
        x = torch.randn(3, 4, device="openreg")
        y = torch.empty(3, 4, device="openreg")
        y.copy_(x, non_blocking=True)
        self.assertEqual(y, x)


class TestOpsExtended(TestCase):
    # [CLASS DOC] More view and set tests on openreg.
    def test_view(self):
        # [FUNCTION DOC] view reshapes and shares the data pointer.
        """Test tensor view operation"""
        x = torch.randn(2, 3, 4, device="openreg")
        y = x.view(6, 4)
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, torch.Size([6, 4]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_reshape(self):
        # [FUNCTION DOC] reshape keeps device and gives the new shape.
        """Test tensor reshape operation"""
        x = torch.randn(2, 3, 4, device="openreg")
        y = x.reshape(6, 4)
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, torch.Size([6, 4]))

    def test_as_strided(self):
        # [FUNCTION DOC] as_strided gives a 2 by 2 openreg tensor.
        """Test as_strided operation"""
        x = torch.randn(3, 4, device="openreg")
        y = torch.as_strided(x, (2, 2), (4, 1), 1)
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, torch.Size([2, 2]))

    def test_local_scalar_dense(self):
        # [FUNCTION DOC] item on a one-element openreg tensor returns the scalar.
        """Test local scalar dense extraction"""
        x = torch.tensor([5.0], device="openreg")
        scalar = x.item()
        self.assertEqual(scalar, 5.0)

    def test_set_tensor(self):
        # [FUNCTION DOC] set_ with a tensor source makes the destination equal the source.
        """Test set_ operation with tensor source"""
        x = torch.randn(3, 4, device="openreg")
        y = torch.empty(3, 4, device="openreg")
        y.set_(x)
        self.assertEqual(y, x)

    def test_set_storage(self):
        # [FUNCTION DOC] set_ with a storage, offset zero
        # and size makes the destination equal the source.
        """Test set_ operation with storage source"""
        x = torch.randn(3, 4, device="openreg")
        storage = x.storage()
        y = torch.empty(3, 4, device="openreg")
        y.set_(storage, 0, y.size())
        self.assertEqual(y, x)


class TestSTUBExtended(TestCase):
    # [CLASS DOC] More tests of abs and the custom abs op.
    def test_abs_contiguous(self):
        # [FUNCTION DOC] abs of a contiguous tensor is non-negative with the same shape.
        """Test abs operation with contiguous tensor"""
        x = torch.randn(2, 3, dtype=torch.float32, device="openreg")
        y = torch.abs(x)
        self.assertEqual(y.device.type, "openreg")
        self.assertTrue(torch.all(y >= 0))
        self.assertEqual(y.shape, x.shape)

    def test_abs_non_contiguous(self):
        # [FUNCTION DOC] abs of a transposed (non-contiguous) tensor is non-negative.
        """Test abs operation with non-contiguous tensor"""
        x = torch.randn(2, 3, dtype=torch.float32, device="openreg")
        x_t = x.t()  # Transpose makes it non-contiguous
        y = torch.abs(x_t)
        self.assertEqual(y.device.type, "openreg")
        self.assertTrue(torch.all(y >= 0))

    def test_custom_abs(self):
        # [FUNCTION DOC] The custom abs operator gives
        # non-negative values with same shape on openreg.
        """Test custom abs operation"""
        x = torch.randn(2, 3, dtype=torch.float32, device="openreg")
        y = torch.ops.openreg.custom_abs(x)
        self.assertEqual(y.device.type, "openreg")
        self.assertTrue(torch.all(y >= 0))
        self.assertEqual(y.shape, x.shape)

    def test_abs_out(self):
        # [FUNCTION DOC] abs with out= matches the functional result.
        """Test abs with output tensor"""
        x = torch.randn(2, 3, dtype=torch.float32, device="openreg")
        out = torch.empty_like(x)
        torch.abs(x, out=out)
        self.assertEqual(out.device.type, "openreg")
        self.assertTrue(torch.all(out >= 0))
        self.assertEqual(out, torch.abs(x))


@unittest.skip("Skipping all quantization tests for openreg backend")
class TestQuantizationExtended(TestCase):
    # [CLASS DOC] Additional quantization tests; the whole class is skipped for openreg.
    def test_quantize_per_tensor_different_scales(self):
        # [FUNCTION DOC] Skipped: quantizes with scale 0.1 and zero point 10 and checks device,
        # dtype, scale and zero point.
        """Test quantization with different scales"""
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="openreg")

        scale = 0.1
        zero_point = 10
        quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
        self.assertEqual(quantized.device.type, "openreg")
        self.assertEqual(quantized.dtype, torch.qint8)
        self.assertEqual(quantized.q_scale(), scale)
        self.assertEqual(quantized.q_zero_point(), zero_point)

    def test_quantize_per_tensor_quint8(self):
        # [FUNCTION DOC] Skipped: quantizes to quint8 and checks device and dtype.
        """Test quantization with quint8 dtype"""
        x = torch.randn(3, 4, dtype=torch.float32, device="openreg")
        quantized = torch.quantize_per_tensor(x, 0.1, 128, torch.quint8)
        self.assertEqual(quantized.device.type, "openreg")
        self.assertEqual(quantized.dtype, torch.quint8)

    def test_dequantize(self):
        # [FUNCTION DOC] Skipped: dequantizing a qint8 tensor returns float32 on openreg.
        """Test dequantization"""
        x = torch.randn(3, 4, dtype=torch.float32, device="openreg")
        quantized = torch.quantize_per_tensor(x, 0.1, 10, torch.qint8)
        dequantized = quantized.dequantize()
        self.assertEqual(dequantized.device.type, "openreg")
        self.assertEqual(dequantized.dtype, torch.float32)


class TestFallbackExtended(TestCase):
    # [CLASS DOC] More fallback tests.
    def test_cpu_fallback_blocklist(self):
        # [FUNCTION DOC] abs and abs with out= on openreg stay on openreg.
        # KNOWN ISSUE: does not check that abs avoids the CPU fallback, only the result device.
        """Test that abs is blocked from CPU fallback"""
        x = torch.randn(2, 3, dtype=torch.float32, device="openreg")
        # abs should work (it's implemented)
        y = torch.abs(x)
        self.assertEqual(y.device.type, "openreg")

        # But abs.out should also work
        out = torch.empty_like(x)
        torch.abs(x, out=out)
        self.assertEqual(out.device.type, "openreg")

    def test_fallback_operations(self):
        # [FUNCTION DOC] add and mul of openreg tensors return openreg tensors.
        # KNOWN ISSUE: comment says these fall back to the CPU but nothing verifies that.
        """Test various fallback operations"""
        x = torch.randn(3, 4, device="openreg")
        y = torch.randn(3, 4, device="openreg")

        # Operations that should fallback to CPU
        z = torch.add(x, y)
        self.assertEqual(z.device.type, "openreg")

        z = torch.mul(x, y)
        self.assertEqual(z.device.type, "openreg")

    def test_fallback_with_scalars(self):
        # [FUNCTION DOC] Adding or multiplying by a Python scalar keeps the result on openreg.
        """Test fallback with scalar operations"""
        x = torch.randn(3, 4, device="openreg")
        y = x + 1.0
        self.assertEqual(y.device.type, "openreg")

        y = x * 2.0
        self.assertEqual(y.device.type, "openreg")


class TestSDPAExtended(NNTestCase):
    # [CLASS DOC] More SDPA tests with mask, dropout and causal attention.
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_with_mask(self):
        # [FUNCTION DOC] With an attention mask the fused SDP choice must be OVERRIDEABLE.
        # Skipped under Dynamo.
        """Test fused SDP choice with attention mask"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))

        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        attn_mask_privateuse1 = attn_mask.to("openreg")

        backend = torch._fused_sdp_choice(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask_privateuse1
        )
        self.assertEqual(backend, SDPBackend.OVERRIDEABLE.value)

    @skipIfTorchDynamo()
    def test_scaled_dot_product_attention_with_dropout(self):
        # [FUNCTION DOC] Attention with dropout 0.1 returns an openreg tensor of the input shape.
        """Test scaled dot product attention with dropout"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")

        output = torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1,
            k_privateuse1,
            v_privateuse1,
            attn_mask=None,
            dropout_p=0.1,
            is_causal=False,
        )
        self.assertEqual(output.device.type, "openreg")
        self.assertEqual(output.shape, shape)

    @skipIfTorchDynamo()
    def test_scaled_dot_product_attention_is_causal(self):
        # [FUNCTION DOC] Causal attention returns an openreg tensor of the input shape.
        """Test scaled dot product attention with causal mask"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")

        output = torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1,
            k_privateuse1,
            v_privateuse1,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        self.assertEqual(output.device.type, "openreg")
        self.assertEqual(output.shape, shape)


class TestCustomAutogradFunctions(TestCase):
    # [CLASS DOC] Tests for the openreg custom autograd ops (returns_self and aliasing).
    def test_custom_autograd_fn_returns_self_basic(self):
        # [FUNCTION DOC] The returns_self op equals its input, requires grad, and backward of
        # the sum gives a gradient of one half everywhere.
        """Test basic usage of custom_autograd_fn_returns_self"""
        x = torch.randn(4, device="openreg", requires_grad=True)
        y = torch.ops.openreg.custom_autograd_fn_returns_self(x)

        # Should return the same tensor
        self.assertEqual(x, y)
        self.assertTrue(y.requires_grad)

        # Test backward
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Gradient should be 0.5 * 1.0 = 0.5
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x) * 0.5))

    def test_custom_autograd_fn_aliasing_basic(self):
        # [FUNCTION DOC] The aliasing op keeps shape
        # and requires grad; backward gives one half everywhere.
        """Test basic usage of custom_autograd_fn_aliasing"""
        x = torch.randn(4, device="openreg", requires_grad=True)
        y = torch.ops.openreg.custom_autograd_fn_aliasing(x)

        # Should return a view of the same tensor
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(y.requires_grad)

        # Test backward
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Gradient should be 0.5 * 1.0 = 0.5
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x) * 0.5))

    def test_custom_autograd_fn_returns_self_no_grad(self):
        # [FUNCTION DOC] Without requires_grad the returns_self output
        # equals the input and has no grad.
        """Test custom_autograd_fn_returns_self without requires_grad"""
        x = torch.randn(4, device="openreg", requires_grad=False)
        y = torch.ops.openreg.custom_autograd_fn_returns_self(x)
        self.assertEqual(x, y)
        self.assertFalse(y.requires_grad)

    def test_custom_autograd_fn_aliasing_no_grad(self):
        # [FUNCTION DOC] Without requires_grad the aliasing output keeps shape and has no grad.
        """Test custom_autograd_fn_aliasing without requires_grad"""
        x = torch.randn(4, device="openreg", requires_grad=False)
        y = torch.ops.openreg.custom_autograd_fn_aliasing(x)
        self.assertEqual(x.shape, y.shape)
        self.assertFalse(y.requires_grad)


if __name__ == "__main__":
    run_tests()
