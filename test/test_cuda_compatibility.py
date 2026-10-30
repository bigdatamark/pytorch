# Owner(s): ["module: cuda"]
# [MODULE DOC] Unit tests for the CUDA compute-capability compatibility helpers in torch.cuda.
# Covers the code-versus-device compatibility rule and the startup capability check that warns
# when no compiled architecture suits an installed GPU. All CUDA state and the version string
# are mocked, so no GPU is needed. No import-time side effects beyond imports.

import warnings
from unittest.mock import patch

import torch
import torch.cuda
from torch.testing._internal.common_utils import run_tests, TestCase


@patch("torch.version.cuda", "12.6")
class TestCodeCompatibleWithDevice(TestCase):
    # [CLASS DOC] Tests for the pairwise compatibility predicate, with the CUDA version string
    # patched to 12.6 for the whole class. Stateless; each test passes integer compute capability
    # values (major * 10 + minor) and checks the boolean result or an emitted warning.
    def test_compatible_cases(self):
        # [FUNCTION DOC] Same capability, and a newer device of the same major with older code,
        # are both reported compatible.
        self.assertTrue(
            torch.cuda._code_compatible_with_device(device_cc=80, code_cc=80)
        )
        self.assertTrue(
            torch.cuda._code_compatible_with_device(device_cc=86, code_cc=80)
        )

    def test_backward_incompatible(self):
        # [FUNCTION DOC] Code built for a higher minor capability than the device is rejected.
        self.assertFalse(
            torch.cuda._code_compatible_with_device(device_cc=80, code_cc=86)
        )

    def test_cross_major_incompatible(self):
        # [FUNCTION DOC] Code is rejected when the device has a different major version, whether
        # the device major is higher or lower than the code major.
        self.assertFalse(
            torch.cuda._code_compatible_with_device(device_cc=90, code_cc=80)
        )
        self.assertFalse(
            torch.cuda._code_compatible_with_device(device_cc=75, code_cc=80)
        )

    def test_igpu_cases_pre_132(self):
        # [FUNCTION DOC] With CUDA 12.6, integrated-GPU capabilities 53 and 87 do not accept the
        # generic 50 or 80 code, but 53 code runs on a 53 device.
        self.assertFalse(
            torch.cuda._code_compatible_with_device(device_cc=53, code_cc=50)
        )
        self.assertFalse(
            torch.cuda._code_compatible_with_device(device_cc=87, code_cc=80)
        )
        self.assertTrue(
            torch.cuda._code_compatible_with_device(device_cc=53, code_cc=53)
        )

    def test_igpu_cases_post_132(self):
        # [FUNCTION DOC] With the version patched to 13.2, an 87 device accepts 80 code and a 53
        # device accepts 53 code.
        with patch("torch.version.cuda", "13.2"):
            self.assertTrue(
                torch.cuda._code_compatible_with_device(device_cc=87, code_cc=80)
            )
            self.assertTrue(
                torch.cuda._code_compatible_with_device(device_cc=53, code_cc=53)
            )

    def test_special_case_sm101_on_sm110(self):
        # [FUNCTION DOC] Code built for capability 101 is accepted on a capability 110 device.
        self.assertTrue(
            torch.cuda._code_compatible_with_device(device_cc=110, code_cc=101)
        )

    def test_unknown_code_cc(self):
        # [FUNCTION DOC] For an unknown code capability (990), the function returns compatible for
        # an equal or higher same-major device and emits exactly one warning mentioning an unknown
        # compute capability. Records warnings in both phases and checks the count in each.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.cuda._code_compatible_with_device(device_cc=990, code_cc=990)
            self.assertTrue(result)
            self.assertEqual(len(w), 1)
            self.assertIn("unknown compute capability", str(w[0].message))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.cuda._code_compatible_with_device(device_cc=991, code_cc=990)
            self.assertTrue(result)
            self.assertEqual(len(w), 1)


@patch("torch.cuda.get_device_name", return_value="NVIDIA MOCK DEVICE")
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.version.cuda", "12.6")
class TestCheckCapability(TestCase):
    # [CLASS DOC] Tests for the startup capability check. The device name, device count (one) and
    # CUDA version (12.6) are patched for the whole class; individual tests patch the architecture
    # list and the reported device capability, then check whether and what warning is emitted.
    def test_rocm_skips_check(self, *args):
        # [FUNCTION DOC] With the CUDA version patched to None (ROCm-like build), the check
        # returns without warning; warnings are turned into errors to prove silence.
        with (
            patch("torch.version.cuda", None),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            self.assertIsNone(torch.version.cuda)
            torch.cuda._check_capability()

    @patch("torch.cuda.get_arch_list", return_value=["sm_70", "sm_80", "sm_90"])
    @patch("torch.cuda.get_device_capability", return_value=(8, 0))
    def test_compatible_device_no_warning(self, *args):
        # [FUNCTION DOC] A capability 8.0 device with a build for sm_70, sm_80 and sm_90 produces
        # no warning (warnings are errors).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            torch.cuda._check_capability()

    @patch("torch.cuda.get_arch_list", return_value=["sm_80"])
    @patch("torch.cuda.get_device_capability", return_value=(7, 5))
    def test_incompatible_device_warns(self, *args):
        # [FUNCTION DOC] A capability 7.5 device with only an sm_80 build raises a warning naming
        # GPU0 and compute capability 7.5.
        with self.assertWarnsRegex(
            UserWarning, r"Found GPU0.*which is of compute capability.*7\.5"
        ):
            torch.cuda._check_capability()

    @patch("torch.cuda.get_arch_list", return_value=["sm_80"])
    @patch("torch.cuda.get_device_capability", return_value=(8, 7))
    def test_incompatible_device_warns_igpu(self, *args):
        # [FUNCTION DOC] A capability 8.7 device with only an sm_80 build raises the unsupported
        # GPU warning for GPU0 (pre-13.2 rules, so integrated GPUs do not accept generic code).
        with self.assertWarnsRegex(
            UserWarning, r"Found GPU0.*which is of compute capability.*8\.7"
        ):
            torch.cuda._check_capability()

    @patch("torch.cuda.get_arch_list", return_value=["sm_80", "sm_90"])
    def test_multiple_devices_mixed_compatibility(self, *args):
        # [FUNCTION DOC] With three mocked devices (8.0, 7.5, 8.6) and a build for sm_80 and
        # sm_90, exactly one warning is emitted and it names GPU1, the only unsupported device.
        caps = [(8, 0), (7, 5), (8, 6)]
        with (
            patch("torch.cuda.device_count", return_value=len(caps)),
            patch("torch.cuda.get_device_capability", side_effect=caps),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            torch.cuda._check_capability()
            self.assertEqual(len(w), 1)
            self.assertIn("GPU1", str(w[0].message))

    @patch("torch.cuda.get_arch_list", return_value=["sm_80", "sm_90"])
    @patch("torch.cuda.get_device_capability", return_value=(7, 5))
    def test_warning_message_contains_device_info(self, *args):
        # [FUNCTION DOC] The warning for a 7.5 device lists the device name, its compute
        # capability, and the build capabilities 8.0 and 9.0 with the hardware each supports.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.cuda._check_capability()
            self.assertEqual(len(w), 1)
            msg = str(w[0].message)
            self.assertIn("GPU0", msg)
            self.assertIn("NVIDIA MOCK DEVICE", msg)
            self.assertIn("compute capability (CC) 7.5", msg)
            self.assertIn("8.0 which supports", msg)
            self.assertIn("9.0 which supports", msg)

    @patch("torch.cuda.get_arch_list", return_value=["sm_60"])
    @patch("torch.cuda.get_device_capability", return_value=(7, 0))
    @patch("torch.cuda._host_arch_key", return_value="x86_64")
    @patch(
        "torch.cuda.PYTORCH_RELEASES_CODE_CC",
        {
            "12.6": {"x86_64": {50, 60, 70}, "aarch64": {50, 60, 70}},
            "12.8": {"x86_64": {70}, "aarch64": {70}},
            "13.2": {"x86_64": {75}, "aarch64": {75}},
        },
    )
    def test_warning_suggests_compatible_pytorch_release(self, *args):
        # [FUNCTION DOC] With a patched release table and an x86_64 host, a 7.0 device and an
        # sm_60 build, the warning suggests the CUDA 12.6 and 12.8 releases but not 13.2.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.cuda._check_capability()
            self.assertEqual(len(w), 1)
            msg = str(w[0].message)
            self.assertIn("12.6", msg)
            self.assertIn("12.8", msg)
            self.assertNotIn("13.2", msg)

    @patch("torch.cuda.get_arch_list", return_value=["sm_80"])
    @patch("torch.cuda.get_device_capability", return_value=(5, 3))
    def test_warning_no_compatible_pytorch_release(self, *args):
        # [FUNCTION DOC] For a 5.3 device with an sm_80 build and no published release supporting
        # it, the warning has no install command, says no published builds support the GPU, and
        # points to the PyTorch get-started page.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.cuda._check_capability()
            self.assertEqual(len(w), 1)
            msg = str(w[0].message)
            self.assertNotIn("pip install torch==", msg)
            self.assertIn("No published PyTorch CUDA builds for release", msg)
            self.assertIn("https://pytorch.org/get-started/locally/", msg)

    @patch("torch.cuda.get_arch_list", return_value=["sm_80"])
    @patch("torch.cuda.get_device_capability", return_value=(8, 7))
    def test_sbsa_runs_on_jetson_post_cuda132(self, *args):
        # [FUNCTION DOC] With the version patched to 13.2, an 8.7 device and an sm_80 build, the
        # check emits no warning (warnings are errors).
        with (
            patch("torch.version.cuda", "13.2"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            torch.cuda._check_capability()

    @patch("torch.cuda.get_arch_list", return_value=["sm_87"])
    @patch("torch.cuda.get_device_capability", return_value=(8, 9))
    def test_jetson_doesnt_run_on_sbsa_post_cuda132(self, *args):
        # [FUNCTION DOC] With the version patched to 13.2, an 8.9 device and an sm_87 build, the
        # check warns for GPU0 at compute capability 8.9.
        with (
            patch("torch.version.cuda", "13.2"),
            self.assertWarnsRegex(
                UserWarning, r"Found GPU0.*which is of compute capability.*8\.9"
            ),
        ):
            torch.cuda._check_capability()


if __name__ == "__main__":
    run_tests()
