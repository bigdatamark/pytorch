# Owner(s): ["oncall: export"]
# [MODULE DOC] Tests for the experimental export package class (exporting functions with fallbacks
# and overloads) and for AOTInductor package loading error handling when the device is unavailable.
# Uses mocks to replace the C++ package loader and the pt2 loader.
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._inductor.package import load_package
from torch.export import Dim
from torch.export.experimental import _ExportPackage
from torch.export.pt2_archive._package import _load_aoti
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not is_dynamo_supported(), "dynamo isn't supported")
class TestPackage(TestCase):
    # [CLASS DOC] Tests of the experimental export package object, skipped when Dynamo is
    # unsupported.
    def test_basic(self):
        # [FUNCTION DOC] Exports a simple add-one function through a package exporter; checks the
        # call result equals eager, that the package has one method, which has one fallback and no
        # overloads.
        def fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Function returning the input plus one.
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        self.assertEqual(
            package._exporter("fn", fn)(x),
            fn(x),
        )
        self.assertEqual(len(package.methods), 1)
        self.assertEqual(len(package.methods["fn"].fallbacks), 1)
        self.assertEqual(len(package.methods["fn"].overloads), 0)

    def test_more_than_once(self):
        # [FUNCTION DOC] Calling the same exporter twice raises a RuntimeError saying it cannot
        # export more than once.
        def fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Function returning the input plus one.
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        exporter = package._exporter("fn", fn)
        exporter(x)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot export .* more than once",
        ):
            exporter(x)

    def test_error(self):
        # [FUNCTION DOC] With the fallback policy set to error, calling the exporter without any
        # overload defined raises a RuntimeError saying the fallback cannot be exported under that
        # policy.
        def fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Function returning the input plus one.
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        exporter = package._exporter("fn", fn, fallback="error")
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot export fallback .* when fallback policy is set to 'error'",
        ):
            exporter(x)

    def test_overloads(self):
        # [FUNCTION DOC] Defines a module whose result depends on the first dimension (three
        # subtracts one, four adds one, otherwise adds two) and registers three overload specs for
        # sizes three, four and at least five (the last returning a dynamic batch dimension
        # constraint). Checks each size gives the matching eager-equivalent result and that the
        # package has one method with three overloads.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a size-dependent branch on the first dimension of its
            # input.
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns input plus one for size four, minus one for size
                # three, otherwise plus two.
                if x.shape[0] == 4:
                    return x + 1
                elif x.shape[0] == 3:
                    return x - 1
                else:
                    return x + 2

        fn = Module()
        x = torch.randn(3, 2)
        x2 = torch.randn(4, 2)
        x3 = torch.randn(5, 2)

        def spec(self, x):
            # [FUNCTION DOC] HELPER: Overload spec that asserts the first dimension is three (its
            # assertion is evaluated when the spec is applied).
            assert x.shape[0] == 3  # noqa: S101

        def spec2(self, x):
            # [FUNCTION DOC] HELPER: Overload spec that asserts the first dimension is four.
            assert x.shape[0] == 4  # noqa: S101

        def spec3(self, x):
            # [FUNCTION DOC] HELPER: Overload spec that asserts the first dimension is at least
            # five and returns a dynamic-shape mapping with a batch dimension with minimum five on
            # the input's first dim and a static second dim.
            assert x.shape[0] >= 5  # noqa: S101
            return {"x": (Dim("batch", min=5), Dim.STATIC)}

        package = _ExportPackage()
        exporter = (
            package._exporter("fn", fn)
            ._define_overload("spec", spec)
            ._define_overload("spec2", spec2)
            ._define_overload("spec3", spec3)
        )
        self.assertEqual(exporter(x), x - 1)
        self.assertEqual(exporter(x2), x2 + 1)
        self.assertEqual(exporter(x3), x3 + 2)
        self.assertEqual(len(package.methods), 1)
        self.assertEqual(len(package.methods["fn"].overloads), 3)


class TestAOTIPackageDeviceValidation(TestCase):
    # [CLASS DOC] Tests that AOTInductor package loading surfaces device-availability errors
    # instead of falling back or probing device information.
    def test_aoti_load_uses_cpp_device_validation_before_device_info(self):
        # [FUNCTION DOC] Replaces the C++ package loader with a fake whose constructor raises a
        # device-unavailable RuntimeError and whose metadata function reports a CUDA device, and
        # patches the device-information lookup. Calling the pt2 AOTI loader must raise that
        # RuntimeError and must not call the device-information lookup.
        class FakeAOTIModelPackageLoader:
            # [CLASS DOC] HELPER: Fake loader class: metadata lookup says CUDA, constructor raises
            # the unavailable-device RuntimeError.
            @staticmethod
            def load_metadata_from_package(file, model_name):
                # [FUNCTION DOC] HELPER: Static; ignores its inputs and returns metadata naming a
                # cuda device.
                return {"AOTI_DEVICE_KEY": "cuda"}

            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError saying the AOTInductor package
                # cannot be loaded because CUDA is unavailable.
                raise RuntimeError(
                    "Cannot load AOTInductor package for device 'cuda' because "
                    "CUDA is not available in this process."
                )

        with (
            mock.patch.object(
                torch._C._aoti, "AOTIModelPackageLoader", FakeAOTIModelPackageLoader
            ),
            mock.patch("torch._inductor.codecache.get_device_information") as get_info,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Cannot load AOTInductor package.*CUDA is not available",
            ):
                _load_aoti("model.pt2", "model", False, 1, -1)

            get_info.assert_not_called()

    def test_load_package_does_not_fallback_on_device_validation_error(self):
        # [FUNCTION DOC] Patches the pt2 loader to raise a device-unavailable RuntimeError and the
        # C++ loader with a fake that would raise AssertionError if constructed. Checks
        # load_package propagates the RuntimeError, i.e. does not fall back to the C++ loader.
        class FakeAOTIModelPackageLoader:
            # [CLASS DOC] HELPER: Fake loader that fails the test with AssertionError if it is ever
            # constructed.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises AssertionError saying the fallback loader
                # should not be constructed.
                raise AssertionError("AOTI fallback loader should not be constructed")

        with (
            mock.patch(
                "torch._inductor.package.package.load_pt2",
                side_effect=RuntimeError(
                    "Cannot load AOTInductor package for device 'cuda' because "
                    "CUDA is not available in this process."
                ),
            ),
            mock.patch.object(
                torch._C._aoti, "AOTIModelPackageLoader", FakeAOTIModelPackageLoader
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Cannot load AOTInductor package.*CUDA is not available",
            ):
                load_package("model.pt2")

    @unittest.skipIf(
        torch.cuda.is_available(), "requires CUDA to be unavailable in this process"
    )
    def test_cpp_aoti_package_loader_validates_cuda_availability(self):
        # [FUNCTION DOC] Creates a temporary directory laid out like an unpacked AOTI package for a
        # model with a placeholder shared library file and metadata naming a cuda device, then
        # checks the real C++ loader raises a RuntimeError about CUDA or ROCm being unavailable.
        # Skipped when CUDA is available.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "data" / "aotinductor" / "model"
            model_dir.mkdir(parents=True)
            extension = ".pyd" if sys.platform == "win32" else ".so"
            (model_dir / f"model{extension}").touch()
            (model_dir / "model_metadata.json").write_text(
                json.dumps({"AOTI_DEVICE_KEY": "cuda"}), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Cannot load AOTInductor package.*(CUDA|ROCm) is not available",
            ):
                torch._C._aoti.AOTIModelPackageLoader(str(tmp), "model", False, 1, -1)


if __name__ == "__main__":
    run_tests()
