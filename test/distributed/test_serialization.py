# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Round-trip tests for the streaming state dict serializer in
# torch.distributed._serialization (_streaming_save and _streaming_load): scalar, strided,
# offset, nested, mixed dtype, empty, DTensor, arbitrary Python object, UTF-8 string,
# weights_only handling and CUDA tensors. Defines the debug env var name constant and a small
# picklable class. No import-time side effects beyond imports.

import os
import pickle
from io import BytesIO
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed._serialization import _streaming_load, _streaming_save
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor
from torch.testing._internal.common_utils import requires_cuda, run_tests, TestCase


DEBUG_ENV = "TORCH_SERIALIZATION_DEBUG"


class MyClass:
    # [CLASS DOC] HELPER: Tiny picklable value class holding one integer attribute a, used as
    # a non-tensor object in serialization tests. Equality compares a only.
    def __init__(self, a: int) -> None:
        # [FUNCTION DOC] HELPER: Store the integer a on the instance.
        self.a = a

    def __eq__(self, other: "MyClass") -> bool:
        # [FUNCTION DOC] HELPER: Return whether the other object's a equals this one's a.
        # Raises AttributeError if other has no a.
        # KNOWN ISSUE: defines equality without a hash, so instances are unhashable.
        return self.a == other.a


class TestSerialization(TestCase):
    # [CLASS DOC] Tests of the streaming save and load round trip. setUp disables the
    # serialization debug asserts through an environment variable and tearDown restores the
    # old value.
    def setUp(self) -> None:
        # [FUNCTION DOC] Run base setUp, remember the debug env var value, then set it to 0 to
        # disable debug asserts.
        super().setUp()
        # disable debug asserts
        self._old_debug = os.environ.get(DEBUG_ENV)
        os.environ[DEBUG_ENV] = "0"

    def tearDown(self):
        # [FUNCTION DOC] Restore the debug env var only if it was set before.
        # KNOWN ISSUE: if it was originally unset the value 0 remains in the environment, and
        # the base tearDown is not called.
        if self._old_debug is not None:
            os.environ[DEBUG_ENV] = self._old_debug

    def test_scalar_tensor(self) -> None:
        # [FUNCTION DOC] Round trip of a state dict with a 0-d int32 tensor: saves a state
        # dict to an in-memory buffer with the streaming save, rewinds it, loads it with the
        # streaming load and compares with assert_close.
        tensor = torch.tensor(42, dtype=torch.int32)
        state_dict = {"scalar": tensor}
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_strided_tensor(self) -> None:
        # [FUNCTION DOC] Round trip of a non-contiguous tensor (every other row and column of
        # a 4 by 4 float tensor): saves a state dict to an in-memory buffer with the streaming
        # save, rewinds it, loads it with the streaming load and compares with assert_close.
        base_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        strided_tensor = base_tensor[::2, ::2]
        state_dict = {"strided": strided_tensor}
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_tensor_with_offset(self) -> None:
        # [FUNCTION DOC] Round trip of float64 tensors that are a sliced view with storage
        # offset and a strided slice of an aranged tensor: saves a state dict to an in-memory
        # buffer with the streaming save, rewinds it, loads it with the streaming load and
        # compares with assert_close.
        state_dict = {
            "offset": torch.arange(10, dtype=torch.float64)[2:],
            "strided": torch.arange(10, dtype=torch.float64)[2::2],
        }
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_nested_tensors(self) -> None:
        # [FUNCTION DOC] Round trip of a nested dict holding an int32 and a float64 tensor:
        # saves a state dict to an in-memory buffer with the streaming save, rewinds it, loads
        # it with the streaming load and compares with assert_close.
        tensor1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        tensor2 = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float64)
        state_dict = {"nested": {"tensor1": tensor1, "tensor2": tensor2}}
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_various_data_types(self) -> None:
        # [FUNCTION DOC] Round trip of float32, int16, bool and uint16 tensors: saves a state
        # dict to an in-memory buffer with the streaming save, rewinds it, loads it with the
        # streaming load and compares with assert_close.
        tensor_float32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        tensor_int16 = torch.tensor([1, 2, 3], dtype=torch.int16)
        tensor_bool = torch.tensor([True, False, True], dtype=torch.bool)
        tensor_uint16 = torch.tensor([True, False, True], dtype=torch.uint16)
        state_dict = {
            "float32": tensor_float32,
            "int16": tensor_int16,
            "bool": tensor_bool,
            "uint16": tensor_uint16,
        }
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_empty_tensor(self) -> None:
        # [FUNCTION DOC] Round trip of an empty (0 by 10) tensor, loaded with weights_only
        # false and compared with assertEqual.
        state_dict = {
            "empty": torch.zeros(0, 10),
        }

        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file, weights_only=False)
        self.assertEqual(result, state_dict)

    def test_dtensor(self) -> None:
        # [FUNCTION DOC] Initializes a single-rank gloo process group (hash store), makes a
        # DTensor on a 1-device CPU mesh with empty placements, saves and loads it, and checks
        # the local tensors and the spec match. Side effect: leaves the default process group
        # initialized.
        # KNOWN ISSUE: the process group is never destroyed.
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )

        device_mesh = DeviceMesh("cpu", 1)
        tensor = torch.randn(4, 4)
        dtensor = distribute_tensor(tensor, device_mesh, [])
        state_dict = dtensor
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = cast(DTensor, _streaming_load(file))
        torch.testing.assert_close(result.to_local(), state_dict.to_local())
        self.assertEqual(result._spec, state_dict._spec)

    def test_python_object(self) -> None:
        # [FUNCTION DOC] Round trip of a state dict holding a MyClass instance, loaded with
        # weights_only false and compared by equality.
        state_dict = {
            "obj": MyClass(42),
        }

        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file, weights_only=False)
        self.assertEqual(result, state_dict)

    def test_str_utf8(self) -> None:
        # [FUNCTION DOC] Round trip of a non-ASCII string value (capital U with umlaut),
        # loaded with default settings.
        state_dict = {
            "obj": "Ü",
        }

        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        self.assertEqual(result, state_dict)

    def test_weights_only(self) -> None:
        # [FUNCTION DOC] Saves a state dict with a custom class. Default loading must raise
        # UnpicklingError saying the global is not allowed, and weights_only true combined
        # with an explicit pickle_module must raise RuntimeError mentioning an explicit
        # pickle_module. The same buffer is reused without rewinding for the second call.
        state_dict = {
            "obj": MyClass(42),
        }

        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        with self.assertRaisesRegex(pickle.UnpicklingError, "not an allowed global"):
            _streaming_load(file)

        with self.assertRaisesRegex(RuntimeError, "explicit pickle_module"):
            _streaming_load(file, weights_only=True, pickle_module=pickle)

    @requires_cuda
    def test_cuda(self) -> None:
        # [FUNCTION DOC] Requires CUDA. Round trip of a scalar float tensor on cuda:0; checks
        # values match and the loaded tensor is on the same device.
        device = torch.device("cuda:0")

        tensor = torch.tensor(42, dtype=torch.float, device=device)
        state_dict = {"scalar": tensor}
        file = BytesIO()
        _streaming_save(state_dict, file)
        file.seek(0)

        result = _streaming_load(file)
        torch.testing.assert_close(result, state_dict)
        self.assertEqual(result["scalar"].device, device)


if __name__ == "__main__":
    run_tests()
