# Owner(s): ["module: PrivateUse1"]
# [MODULE DOC] Tests for the PrivateUse1 backend registration utilities as applied to
# openreg: backend renaming, enum aliases, device module registration, generated
# tensor, storage, module and PackedSequence methods, and legacy tensor type strings.
# No import-time side effects.

import types
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestBackendModule(TestCase):
    # [CLASS DOC] Tests for backend name, enum aliases and device module lookup and registration.
    def test_backend_module_name(self):
        # [FUNCTION DOC] Backend name is openreg; renaming to the same name is allowed; renaming
        # to another name raises RuntimeError "has already been set".
        """Test backend module name query and renaming"""
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "openreg")
        # backend can be renamed to the same name multiple times
        torch.utils.rename_privateuse1_backend("openreg")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dev")

    def test_profiler_activity_alias(self):
        # [FUNCTION DOC] ProfilerActivity has an OPENREG alias equal to PrivateUse1 with the
        # expected repr, str and name.
        """Test ProfilerActivity exposes device-specific alias for PrivateUse1"""
        from torch._C._profiler import ProfilerActivity

        self.assertTrue(hasattr(ProfilerActivity, "OPENREG"))
        self.assertEqual(ProfilerActivity.OPENREG, ProfilerActivity.PrivateUse1)
        self.assertEqual(
            repr(ProfilerActivity.OPENREG), "<ProfilerActivity.OPENREG: 5>"
        )
        self.assertEqual(str(ProfilerActivity.OPENREG), "ProfilerActivity.OPENREG")
        self.assertEqual(ProfilerActivity.OPENREG.name, "OPENREG")

    def test_device_type_alias(self):
        # [FUNCTION DOC] DeviceType has an OPENREG alias equal to PrivateUse1 with the expected
        # repr, str and name.
        """Test DeviceType exposes device-specific alias for PrivateUse1"""
        from torch._C._autograd import DeviceType

        self.assertTrue(hasattr(DeviceType, "OPENREG"))
        self.assertEqual(DeviceType.OPENREG, DeviceType.PrivateUse1)
        self.assertEqual(repr(DeviceType.OPENREG), "<DeviceType.OPENREG: 20>")
        self.assertEqual(str(DeviceType.OPENREG), "DeviceType.OPENREG")
        self.assertEqual(DeviceType.OPENREG.name, "OPENREG")

    def test_backend_module_registration(self):
        # [FUNCTION DOC] Registering a fake module under an unknown name or over the existing
        # openreg name raises RuntimeError.
        """Test backend module registration error handling"""

        def generate_faked_module():
            # [FUNCTION DOC] HELPER: Returns a new empty module named fake_module.
            return types.ModuleType("fake_module")

        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            torch._register_device_module("dev", generate_faked_module())
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("openreg", generate_faked_module())

    def test_backend_module_function(self):
        # [FUNCTION DOC] Looking up an unknown custom module function raises RuntimeError; the
        # device_count function is found and returns two.
        """Test backend module function access"""
        with self.assertRaisesRegex(RuntimeError, "Try to call torch.openreg"):
            torch.utils.backend_registration._get_custom_mod_func("func_name_")
        self.assertTrue(
            torch.utils.backend_registration._get_custom_mod_func("device_count")() == 2
        )

    def test_backend_module_function_error_handling(self):
        # [FUNCTION DOC] An unknown function name raises RuntimeError; device_count is found.
        """Test error handling for backend module functions"""
        # Test non-existent function
        with self.assertRaisesRegex(RuntimeError, "Try to call torch.openreg"):
            torch.utils.backend_registration._get_custom_mod_func("non_existent_func")

        # Test valid function
        device_count = torch.utils.backend_registration._get_custom_mod_func(
            "device_count"
        )
        self.assertIsNotNone(device_count)


class TestBackendProperty(TestCase):
    # [CLASS DOC] Tests for methods and properties generated onto Tensor, storage, Module
    # and PackedSequence for the openreg backend.
    def test_backend_generate_methods(self):
        # [FUNCTION DOC] Regenerating methods raises RuntimeError (custom module already
        # there); the generated is_openreg and openreg attributes exist on the five classes.
        """Test backend method generation"""
        with self.assertRaisesRegex(RuntimeError, "The custom device module of"):
            torch.utils.generate_methods_for_privateuse1_backend()

        self.assertTrue(hasattr(torch.Tensor, "is_openreg"))
        self.assertTrue(hasattr(torch.Tensor, "openreg"))
        self.assertTrue(hasattr(torch.TypedStorage, "is_openreg"))
        self.assertTrue(hasattr(torch.TypedStorage, "openreg"))
        self.assertTrue(hasattr(torch.UntypedStorage, "is_openreg"))
        self.assertTrue(hasattr(torch.UntypedStorage, "openreg"))
        self.assertTrue(hasattr(torch.nn.Module, "openreg"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "is_openreg"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "openreg"))

    def test_backend_tensor_methods(self):
        # [FUNCTION DOC] The generated openreg method on a CPU tensor accepts a device object,
        # an indexed device string or an int and returns an openreg tensor.
        """Test backend tensor methods"""
        x = torch.empty(4, 4)
        self.assertFalse(x.is_openreg)

        y = x.openreg(torch.device("openreg"))
        self.assertTrue(y.is_openreg)
        z = x.openreg(torch.device("openreg:0"))
        self.assertTrue(z.is_openreg)
        n = x.openreg(0)
        self.assertTrue(n.is_openreg)

    @unittest.skip("Need to support Parameter in openreg")
    def test_backend_module_methods(self):
        # [FUNCTION DOC] Skipped (Parameter not supported): a module's openreg method should
        # move its parameter from CPU to openreg.
        """Test backend module methods (currently skipped)"""

        class FakeModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one random 3 by 3 parameter.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the module with parameter x.
                super().__init__()
                self.x = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self):
                # [FUNCTION DOC] HELPER: No-op forward.
                pass

        module = FakeModule()
        self.assertEqual(module.x.device.type, "cpu")
        module.openreg()  # type: ignore[misc]
        self.assertEqual(module.x.device.type, "openreg")

    @unittest.skip("Need to support untyped_storage in openreg")
    def test_backend_storage_methods(self):
        # [FUNCTION DOC] Skipped (untyped_storage unsupported): typed and untyped CPU storage
        # should convert to openreg storage via the openreg method.
        """Test backend storage methods (currently skipped)"""
        x = torch.empty(4, 4)

        x_cpu = x.storage()
        self.assertFalse(x_cpu.is_openreg)
        x_openreg = x_cpu.openreg()
        self.assertTrue(x_openreg.is_openreg)

        y = torch.empty(4, 4)

        y_cpu = y.untyped_storage()
        self.assertFalse(y_cpu.is_openreg)
        y_openreg = y_cpu.openreg()
        self.assertTrue(y_openreg.is_openreg)

    def test_backend_packed_sequence_methods(self):
        # [FUNCTION DOC] A CPU PackedSequence is not openreg and its openreg method returns an
        # openreg one.
        """Test backend PackedSequence methods"""
        x = torch.rand(5, 3)
        y = torch.tensor([1, 1, 1, 1, 1])

        z_cpu = torch.nn.utils.rnn.PackedSequence(x, y)
        self.assertFalse(z_cpu.is_openreg)

        z_openreg = z_cpu.openreg()
        self.assertTrue(z_openreg.is_openreg)

    def test_backend_packed_sequence_properties(self):
        # [FUNCTION DOC] Same as the methods test plus checking the data member is on openreg.
        # KNOWN ISSUE: mostly duplicates test_backend_packed_sequence_methods.
        """Test PackedSequence backend properties"""
        x = torch.rand(5, 3)
        y = torch.tensor([1, 1, 1, 1, 1])

        z_cpu = torch.nn.utils.rnn.PackedSequence(x, y)
        self.assertFalse(z_cpu.is_openreg)

        z_openreg = z_cpu.openreg()
        self.assertTrue(z_openreg.is_openreg)

        # Test that data is on correct device
        self.assertTrue(z_openreg.data.is_openreg)

    def test_backend_tensor_methods_different_devices(self):
        # [FUNCTION DOC] The openreg method with indices zero and one gives matching device
        # indices; a plain device object also gives an openreg tensor.
        """Test tensor methods with different device indices"""
        x = torch.empty(4, 4)

        y0 = x.openreg(0)
        self.assertTrue(y0.is_openreg)
        self.assertEqual(y0.device.index, 0)

        y1 = x.openreg(1)
        self.assertTrue(y1.is_openreg)
        self.assertEqual(y1.device.index, 1)

        y_none = x.openreg(torch.device("openreg"))
        self.assertTrue(y_none.is_openreg)


class TestTensorType(TestCase):
    # [CLASS DOC] Tests for legacy tensor and storage type strings of openreg.
    def test_backend_tensor_type(self):
        # [FUNCTION DOC] For nine dtypes, the type string of an empty openreg tensor equals the
        # expected legacy openreg tensor class name.
        # KNOWN ISSUE: loop variable named str shadows the builtin.
        """Test tensor type string representation for different dtypes"""
        dtypes_map = {
            torch.bool: "torch.openreg.BoolTensor",
            torch.double: "torch.openreg.DoubleTensor",
            torch.float32: "torch.openreg.FloatTensor",
            torch.half: "torch.openreg.HalfTensor",
            torch.int32: "torch.openreg.IntTensor",
            torch.int64: "torch.openreg.LongTensor",
            torch.int8: "torch.openreg.CharTensor",
            torch.short: "torch.openreg.ShortTensor",
            torch.uint8: "torch.openreg.ByteTensor",
        }

        for dtype, str in dtypes_map.items():
            x = torch.empty(4, 4, dtype=dtype, device="openreg")
            self.assertTrue(x.type() == str)

    # Note that all dtype-d Tensor objects here are only for legacy reasons
    # and should NOT be used.
    @skipIfTorchDynamo()
    def test_backend_type_methods(self):
        # [FUNCTION DOC] Checks type strings for CPU and openreg tensors and storages; then
        # temporarily sets a fake FloatStorage on the openreg module so the float storage type
        # string changes while int storage stays generic; the finally block resets it to None.
        # Skipped under Dynamo.
        """Test backend type methods for tensor and storage"""
        # Tensor
        tensor_cpu = torch.randn([8]).float()
        self.assertEqual(tensor_cpu.type(), "torch.FloatTensor")

        tensor_openreg = tensor_cpu.openreg()
        self.assertEqual(tensor_openreg.type(), "torch.openreg.FloatTensor")

        # Storage
        storage_cpu = tensor_cpu.storage()
        self.assertEqual(storage_cpu.type(), "torch.FloatStorage")

        tensor_openreg = tensor_cpu.openreg()
        storage_openreg = tensor_openreg.storage()
        self.assertEqual(storage_openreg.type(), "torch.storage.TypedStorage")

        class CustomFloatStorage:
            # [CLASS DOC] HELPER: Stub whose module and name properties mimic a FloatStorage
            # class of the openreg backend.
            @property
            def __module__(self):
                # [FUNCTION DOC] HELPER: Property returning the
                # module path "torch." plus the backend name.
                return "torch." + torch._C._get_privateuse1_backend_name()

            @property
            def __name__(self):
                # [FUNCTION DOC] HELPER: Property returning the string FloatStorage.
                return "FloatStorage"

        try:
            torch.openreg.FloatStorage = CustomFloatStorage()
            self.assertEqual(storage_openreg.type(), "torch.openreg.FloatStorage")

            # test custom int storage after defining FloatStorage
            tensor_openreg = tensor_cpu.int().openreg()
            storage_openreg = tensor_openreg.storage()
            self.assertEqual(storage_openreg.type(), "torch.storage.TypedStorage")
        finally:
            torch.openreg.FloatStorage = None

    def test_backend_storage_type_consistency(self):
        # [FUNCTION DOC] Storage of an openreg tensor reports openreg and its size equals the
        # tensor element count.
        """Test storage type consistency"""
        tensor = torch.randn(4, 4, device="openreg")
        storage = tensor.storage()

        # Storage should be on same device
        self.assertTrue(storage.is_openreg)

        # Test storage size
        self.assertEqual(storage.size(), tensor.numel())


if __name__ == "__main__":
    run_tests()
