# Owner(s): ["module: mtia"]
# [MODULE DOC] Tests generic Stream and Event objects against the fake MTIA device backend
# provided by the JIT-compiled C++ extension cpp_extensions/mtia_extension.cpp. Importing computes
# TEST_ROCM and narrows TEST_CUDA to also require CUDA_HOME; the test class is skipped on non-
# Linux, ARM64, or when any of CUDA, XPU, MPS, PrivateUse1 or ROCm backends is present.

import os
import tempfile
import unittest

import torch
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_LINUX,
    skipIfTorchDynamo,
    TEST_CUDA,
    TEST_MPS,
    TEST_PRIVATEUSE1,
    TEST_XPU,
)
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


# define TEST_ROCM before changing TEST_CUDA
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None
TEST_CUDA = TEST_CUDA and CUDA_HOME is not None


# Since we use a fake MTIA device backend to test generic Stream/Event, device backends are mutual exclusive to each other.
# The test will be skipped if any of the following conditions are met:
@unittest.skipIf(
    IS_ARM64
    or not IS_LINUX
    or TEST_CUDA
    or TEST_XPU
    or TEST_MPS
    or TEST_PRIVATEUSE1
    or TEST_ROCM,
    "Only on linux platform and mutual exclusive to other backends",
)
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionStreamAndEvent(common.TestCase):
    # [CLASS DOC] Test case that compiles the fake MTIA extension once per class and then
    # exercises generic stream and event APIs on it. The class attribute module holds the loaded
    # extension handle. Each test runs with the working directory set to this file's folder.
    """Tests Stream and Event with C++ extensions."""

    module = None

    def setUp(self):
        # [FUNCTION DOC] Saves the current working directory in an instance attribute and changes
        # into the directory containing this test file, so relative paths used by the extension
        # build work.
        super().setUp()
        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def tearDown(self):
        # [FUNCTION DOC] Runs the base teardown, then restores the working directory
        # saved by setUp.
        super().tearDown()
        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class-level cleanup: removes the shared cpp extensions build
        # root directory.
        torch.testing._internal.common_utils.remove_cpp_extensions_build_root()

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class-level setup: removes any stale extension build root,
        # creates a fresh temporary build directory, and JIT-compiles and loads
        # mtia_extension.cpp using an absolute source path derived from this file's
        # location, with extra include paths including two awkward ones (spaces, a
        # quote). Loaded with the Python-module flag off, so it only registers the fake
        # device backend. Stores the result in the class attribute module.
        torch.testing._internal.common_utils.remove_cpp_extensions_build_root()
        build_dir = tempfile.mkdtemp()
        # Load the fake device guard impl.
        src = f"{os.path.abspath(os.path.dirname(__file__))}/cpp_extensions/mtia_extension.cpp"
        cls.module = torch.utils.cpp_extension.load(
            name="mtia_extension",
            sources=[src],
            build_directory=build_dir,
            extra_include_paths=[
                "cpp_extensions",
                "path / with spaces in it",
                "path with quote'",
            ],
            is_python_module=False,
            verbose=True,
        )

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_stream_event(self):
        # [FUNCTION DOC] Exercises the generic Stream and Event classes on the fake MTIA
        # backend. Checks the default stream device type, that a fresh event has event
        # id 0 and the mtia device type, that recording an event on a stream gives it a
        # nonzero id, that an event returned by record_event differs from the first and
        # was made without timing, and that elapsed time between two timing events can
        # be computed. Asking for elapsed time with an event lacking timing raises
        # ValueError with a message about enable_timing. Re-recording an event on the
        # stream keeps its event id unchanged. Prints events and timing for debugging.
        s = torch.Stream()
        self.assertTrue(s.device_type, int(torch._C._autograd.DeviceType.MTIA))
        e = torch.Event(enable_timing=True)
        e1 = torch.Event(enable_timing=True)
        e1.record()
        self.assertTrue(e.device.type, "mtia")
        # Should be nullptr by default
        self.assertTrue(e.event_id == 0)
        s.record_event(event=e)
        print(f"recorded event 1: {e}")
        self.assertTrue(e.event_id != 0)
        # The enable_timing of event created by record_event() is false
        e2 = s.record_event()
        print(f"recorded event 2: {e2}")
        self.assertTrue(e2.event_id != 0)
        self.assertTrue(e2.event_id != e.event_id)
        e.synchronize()
        e1.synchronize()
        e2.synchronize()
        time_elapsed = e.elapsed_time(e1)
        print(f"time elapsed between e and e1: {time_elapsed}")
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            time_elapsed = e.elapsed_time(e2)
        old_event_id = e.event_id
        e.record(stream=s)
        print(f"recorded event 1: {e}")
        self.assertTrue(e.event_id == old_event_id)


if __name__ == "__main__":
    common.run_tests()
