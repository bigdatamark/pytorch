# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests Inductor's open device registration using a C++ extension backend: builds a
# custom cpp device extension, registers a scheduling class and wrapper codegen classes for a
# private-use device, and checks generated code and numerics. Import time: skips or exits
# cleanly if the test_torchinductor module raises SkipTest; defines shortcuts to its helpers.
import os
import sys
import tempfile
import unittest

import torch
import torch._dynamo
import torch.utils.cpp_extension
from torch._C import FileCheck
from torch.testing._internal.common_utils import skipIfWindows
from torch.testing._internal.inductor_utils import has_cpp_wrapper_for_device


try:
    from extension_backends.cpp.extension_codegen_backend import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:extension_codegen_backend
        ExtensionCppWrapperCodegen,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )
except ImportError:
    from .extension_backends.cpp.extension_codegen_backend import (
        ExtensionCppWrapperCodegen,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )

import torch._inductor.config as config
from torch._inductor import cpu_vec_isa, metrics
from torch._inductor.codegen import cpp_utils
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    register_backend_for_device,
    register_device_op_overrides,
)
from torch._inductor.codegen.cpu_device_op_overrides import CpuDeviceOpOverrides
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    parametrize,
    xfailIfS390X,
)


try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code
TestCase = test_torchinductor.TestCase


@xfailIfS390X
class BaseExtensionBackendTests(TestCase):
    # [CLASS DOC] Base class that compiles the extension_device C++ extension once per class
    # into a temporary build directory and, per test, resets Dynamo, switches the working
    # directory to this file's folder (extension sources use relative paths) and undoes the
    # private-use backend renaming at teardown. Class attribute module holds the loaded
    # extension. Marked expected failure on s390x.
    module = None

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: creates a temporary build directory and loads the C++
        # extension device source from the extension_backends folder next to this file, with
        # debug flags and verbose output, storing the loaded module on the class.
        super().setUpClass()

        cls._build_dir = tempfile.TemporaryDirectory()
        source_file_path = os.path.dirname(os.path.abspath(__file__))
        source_file = os.path.join(
            source_file_path, "extension_backends/cpp/extension_device.cpp"
        )
        cls.module = torch.utils.cpp_extension.load(
            name="extension_device",
            sources=[
                str(source_file),
            ],
            extra_cflags=["-g"],
            verbose=True,
            build_directory=cls._build_dir.name,
        )

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: closes the exit stack from the base test case, calls
        # the parent teardown, then removes the temporary build directory.
        cls._stack.close()
        super().tearDownClass()

        cls._build_dir.cleanup()

    def setUp(self):
        # [FUNCTION DOC] Per-test setup: resets Dynamo, runs parent setup, remembers the current
        # working directory and changes to this file's directory; raises AssertionError if the
        # extension module failed to load.
        torch._dynamo.reset()
        super().setUp()

        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        if self.module is None:
            raise AssertionError

    def tearDown(self):
        # [FUNCTION DOC] Per-test teardown: parent teardown, Dynamo reset, deletes the attribute
        # and the sys.modules entry for the renamed private-use backend module if present, then
        # restores the original working directory.
        super().tearDown()
        torch._dynamo.reset()

        backend_name = torch._C._get_privateuse1_backend_name()
        if hasattr(torch, backend_name):
            delattr(torch, backend_name)
        if f"torch.{backend_name}" in sys.modules:
            del sys.modules[f"torch.{backend_name}"]

        os.chdir(self.old_working_dir)


@unittest.skipIf(IS_FBCODE, "cpp_extension doesn't work in fbcode right now")
class ExtensionBackendTests(BaseExtensionBackendTests):
    # [CLASS DOC] Extension backend tests, skipped in fbcode where cpp extensions do not work.
    # Parametrized tests are instantiated after the class definition.
    @skipIfWindows
    def test_open_device_registration(self):
        # [FUNCTION DOC] Renames the private-use backend to extension_device, registers the
        # extension module, and registers scheduling, python wrapper and cpp wrapper codegen
        # classes plus cpu device op overrides; asserts the lookups return them. Creates tensors
        # on the custom device, checks the custom op was invoked, then for cpp_wrapper both on
        # and off compiles a multiply-add function, checks the generated cpp code with FileCheck
        # (a vectorized load or a scalar load, and the device name) and that results equal the
        # cpu reference of 5. Skipped on Windows. Mutates global registries and the device-to-
        # ATen mapping.
        torch.utils.rename_privateuse1_backend("extension_device")
        torch._register_device_module("extension_device", self.module)

        register_backend_for_device(
            "extension_device",
            ExtensionScheduling,
            ExtensionWrapperCodegen,
            ExtensionCppWrapperCodegen,
        )
        register_device_op_overrides("extension_device", CpuDeviceOpOverrides())
        self.assertTrue(
            get_scheduling_for_device("extension_device") == ExtensionScheduling
        )
        self.assertTrue(
            get_wrapper_codegen_for_device("extension_device")
            == ExtensionWrapperCodegen
        )
        self.assertTrue(
            get_wrapper_codegen_for_device("extension_device", True)
            == ExtensionCppWrapperCodegen
        )

        self.assertFalse(self.module.custom_op_called())
        device = self.module.custom_device()
        x = torch.empty(2, 16).to(device=device).fill_(1)
        self.assertTrue(self.module.custom_op_called())
        y = torch.empty(2, 16).to(device=device).fill_(2)
        z = torch.empty(2, 16).to(device=device).fill_(3)
        ref = torch.empty(2, 16).fill_(5)

        self.assertTrue(x.device == device)
        self.assertTrue(y.device == device)
        self.assertTrue(z.device == device)

        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Computes a times b plus c; the function compiled in the
            # test.
            return a * b + c

        cpp_utils.DEVICE_TO_ATEN["extension_device"] = "at::kPrivateUse1"
        for cpp_wrapper_flag in [True, False]:
            with config.patch({"cpp_wrapper": cpp_wrapper_flag}):
                metrics.reset()
                opt_fn = torch.compile()(fn)
                _, code = run_and_get_cpp_code(opt_fn, x, y, z)
                if (
                    cpu_vec_isa.valid_vec_isa_list()
                    and os.getenv("ATEN_CPU_CAPABILITY") != "default"
                ):
                    load_expr = "loadu"
                else:
                    load_expr = " = in_ptr0[static_cast<int64_t>(x0)];"
                FileCheck().check("void").check(load_expr).check(
                    "extension_device"
                ).run(code)
                opt_fn(x, y, z)
                res = opt_fn(x, y, z)
                self.assertEqual(ref, res.to(device="cpu"))

    @parametrize("device", ["cpu", "has_cpp_wrapper_extension_device"])
    def test_has_cpp_wrapper_for_device(self, device: str):
        # [FUNCTION DOC] Parametrized over cpu and a fictitious device name. Calls the lookup
        # with nothing registered (must not raise), registers a backend without cpp wrapper and
        # asserts false, then registers with a cpp wrapper and asserts true. Mutates the global
        # backend registry.
        # Check that calling the function without having registered a backend
        # ourselves doesn't error.
        _ = has_cpp_wrapper_for_device(device)

        # Check when we don't have a C++ wrapper
        register_backend_for_device(
            device,
            ExtensionScheduling,
            ExtensionWrapperCodegen,
        )
        self.assertFalse(has_cpp_wrapper_for_device(device))

        # Check when we have a C++ wrapper
        register_backend_for_device(
            device,
            ExtensionScheduling,
            ExtensionWrapperCodegen,
            ExtensionCppWrapperCodegen,
        )
        self.assertTrue(has_cpp_wrapper_for_device(device))


instantiate_parametrized_tests(ExtensionBackendTests)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    # cpp_extension doesn't work in fbcode right now
    if HAS_CPU and not IS_MACOS and not IS_FBCODE:
        run_tests()
