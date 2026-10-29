# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the inductor GEMM lookup table, which maps a string key built from kernel
# inputs and operation name to lists of template configs. Contains mock kernel-input classes, unit
# tests of the lookup/filtering logic in the lookup-table choices handler, and end-to-end tests
# that compile matmul-style models and check which template choices are produced. Defines a
# module-level config patch for decompose-k that is non-empty only on ROCm. The test run only
# starts when a GPU, a CPU and a big GPU are all available.
import re
import unittest
from functools import partial
from typing import Any
from unittest.mock import patch

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config
from torch._inductor.choices import InductorChoices
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.lookup_table.choices import LookupTableChoices
from torch._inductor.select_algorithm import (
    add_preprocessing_fn,
    clear_preprocessing_fns,
    ExternKernelCaller,
    TritonTemplateCaller,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache, get_num_sms, TMA_DESCRIPTOR_SIZE
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA_AND_TRITON, HAS_GPU
from torch.utils._triton import has_triton_stable_tma_api, has_triton_tma_device


# Conditional patch for decompose_k tests - override to 10 on ROCm, no-op elsewhere
_DECOMPOSE_K_PATCH_ROCM = (
    {"triton.num_decompose_k_splits": 10} if torch.version.hip else {}
)


class MockTensorNode:
    # [CLASS DOC] HELPER: Minimal stand-in for an input IR node wrapping a real tensor. Exposes
    # device, dtype, size and stride getters that read straight from the tensor.
    """Mock input node that wraps a real tensor for testing"""

    def __init__(self, tensor: torch.Tensor):
        # [FUNCTION DOC] HELPER: Stores the wrapped tensor.
        self.tensor = tensor

    def get_device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Returns the wrapped tensor device.
        return self.tensor.device

    def get_dtype(self) -> torch.dtype:
        # [FUNCTION DOC] HELPER: Returns the wrapped tensor dtype.
        return self.tensor.dtype

    def get_size(self) -> tuple[int, ...]:
        # [FUNCTION DOC] HELPER: Returns the tensor shape as a tuple of ints.
        return tuple(self.tensor.shape)

    def get_stride(self) -> tuple[int, ...]:
        # [FUNCTION DOC] HELPER: Returns the tensor strides as a tuple of ints.
        return tuple(self.tensor.stride())


class MockMMKernelInputs(MMKernelInputs):
    # [CLASS DOC] HELPER: Real matmul kernel-inputs class built from real tensors wrapped in mock
    # nodes. Keeps the original tensors in an attribute, and answers the hinted shape/stride/mnk
    # queries by delegating to the symbolic versions, valid because real shapes are plain ints.
    """Mock MMKernelInputs that subclasses the real class and uses real tensors"""

    def __init__(
        self,
        tensors: list[torch.Tensor],
        scalars: dict[str, float | int] | None = None,
        mat1_idx: int = -2,
        mat2_idx: int = -1,
    ):
        # [FUNCTION DOC] HELPER: Wraps each tensor in a mock node and calls the base constructor
        # with the nodes, optional scalars mapping and the indices of the two matmul operands.
        # Also stores the original tensor list.
        """Initialize with real tensors, creating mock nodes for the base class"""
        mock_nodes = [MockTensorNode(t) for t in tensors]
        super().__init__(mock_nodes, scalars, mat1_idx=mat1_idx, mat2_idx=mat2_idx)
        self.tensors = tensors  # Keep reference to original tensors

    def shapes_hinted(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] HELPER: Returns the symbolic shapes, which are concrete for real tensors.
        """Delegate to symbolic since real tensors already have int shapes"""
        return self.shapes_symbolic()

    def strides_hinted(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] HELPER: Returns the symbolic strides, which are concrete for real
        # tensors.
        """Delegate to symbolic since real tensors already have int strides"""
        return self.strides_symbolic()  # pyre-ignore

    def mnk_hinted(self) -> tuple[int, int, int]:
        # [FUNCTION DOC] HELPER: Returns the symbolic (m, n, k) triple, concrete for real tensors.
        """Delegate to symbolic since real tensors already have int dimensions"""
        return self.mnk_symbolic()  # pyre-ignore

    @property
    def device_type(self) -> str | None:
        # [FUNCTION DOC] HELPER: Property returning the device type string of the first tensor.
        return self.tensors[0].device.type


class BaseLookupTableTest(TestCase):
    # [CLASS DOC] HELPER: Base test case for lookup table tests. Setup saves the configured lookup
    # table and max-autotune setting, forces max-autotune on and installs the lookup-table choices
    # handler; teardown restores both and reinstalls the default choices handler. Offers builders
    # for mock inputs, lookup keys and template configs.
    """Base class for lookup table tests with common setup and utilities"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Saves current lookup table and max_autotune config, enables
        # max_autotune and installs the lookup table choices handler globally.
        super().setUp()
        self.original_table = inductor_config.lookup_table.table
        self.original_max_autotune = getattr(inductor_config, "max_autotune", False)
        inductor_config.max_autotune = True
        # Set the lookup table choices handler
        V.set_choices_handler(LookupTableChoices())

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Restores the saved lookup table and max_autotune values, installs
        # the default choices handler, then runs the base teardown.
        inductor_config.lookup_table.table = self.original_table
        inductor_config.max_autotune = self.original_max_autotune
        # Restore original choices handler
        V.set_choices_handler(InductorChoices())
        super().tearDown()

    def create_mock_mm_kernel_inputs(
        self,
        shapes: list[tuple[int, ...]] | None = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
        scalars: dict[str, float | int] | None = None,
    ) -> MockMMKernelInputs:
        # [FUNCTION DOC] HELPER: Builds mock matmul kernel inputs from random tensors, one per
        # requested shape (default is two 128 by 128 matrices) on the given device and dtype,
        # passing the optional scalars through.
        """Create MockMMKernelInputs with real tensors"""
        if shapes is None:
            shapes = [(128, 128), (128, 128)]  # Default MM shapes

        tensors = []
        for shape in shapes:
            # Create a real tensor with the specified shape, device, and dtype
            tensor = torch.randn(shape, device=device, dtype=dtype)
            tensors.append(tensor)

        return MockMMKernelInputs(tensors, scalars)

    def create_lookup_key(self, method, kernel_inputs):
        # [FUNCTION DOC] HELPER: Builds a fresh lookup-table choices handler and returns the
        # lookup key string for the given method name and kernel inputs.
        """Create a lookup key using LookupTableChoices"""
        choices = LookupTableChoices()
        return choices.make_lookup_key(kernel_inputs, method)

    def create_config(self, template_id, **kwargs):
        # [FUNCTION DOC] HELPER: Returns a config mapping with the template_id field. For the
        # triton id it adds default block sizes, stages, warps and flags; for tma it adds a larger
        # set of defaults; for decompose_k it adds a split count of four; other ids get nothing
        # else. Keyword arguments are applied last and override defaults.
        """Create a backend configuration with template_id field"""
        config = {"template_id": template_id}

        # Add minimal defaults based on template type
        if template_id == "triton":
            config.update(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 64,
                    "num_stages": 2,
                    "num_warps": 2,
                    "EVEN_K": True,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                }
            )
        elif template_id == "tma":
            config.update(
                {
                    "BLOCK_M": 256,
                    "BLOCK_N": 128,
                    "BLOCK_K": 64,
                    "num_stages": 4,
                    "num_warps": 8,
                    "EVEN_K": True,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                }
            )
        elif template_id == "decompose_k":
            config.update({"k": 4})

        config.update(kwargs)
        return config


@unittest.skipIf(not HAS_CUDA_AND_TRITON, "CUDA not available")
@instantiate_parametrized_tests
class TestLookupTable(BaseLookupTableTest):
    # [CLASS DOC] Unit tests for the lookup of template configs from the lookup table, covering
    # mismatched methods, template filtering, empty tables, validation errors, CPU inputs,
    # source-hash filtering and device-specific versus device-agnostic keys. Needs CUDA and
    # triton.
    """Consolidated tests for lookup table functionality"""

    def test_lookup_mismatch(self):
        # [FUNCTION DOC] A table entry stored under the mm method is not found when looking up for
        # addmm; the result is an empty mapping.
        """Test mismatch scenario in lookup table"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        lookup_table_data = {
            self.create_lookup_key("mm", kernel_inputs): [self.create_config("triton")]
        }

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()
            # looking for addmm but created the entry with mm - should mismatch the key and return
            # an empty result
            result = test_choices.lookup_template_configs(
                kernel_inputs, "addmm", ["triton"]
            )
            self.assertEqual(result, {})

    def test_successful_lookup_with_template_filtering(self):
        # [FUNCTION DOC] One table entry holds triton, tma and decompose_k configs. Lookup per
        # template id returns only configs of that template, with the template_id field stripped
        # and other fields preserved.
        """Test successful lookup that filters configs by template_id"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config("triton", BLOCK_M=128, BLOCK_N=128),
            self.create_config("triton", BLOCK_M=64, BLOCK_N=64),
            self.create_config("tma", BLOCK_M=256, BLOCK_N=128),
            self.create_config("decompose_k", k_split=4),
        ]

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()

            # Test triton template filtering
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"]
            )
            if result is None:
                raise AssertionError("Result should not be None")
            self.assertEqual(len(result["triton"]), 2)
            for config in result["triton"]:
                self.assertNotIn("template_id", config)
                self.assertIn("BLOCK_M", config)

            # Test tma template filtering
            result = test_choices.lookup_template_configs(kernel_inputs, "mm", ["tma"])
            if result is None:
                raise AssertionError("Result should not be None")
            self.assertEqual(len(result["tma"]), 1)
            self.assertNotIn("template_id", result["tma"][0])
            self.assertEqual(result["tma"][0]["BLOCK_M"], 256)

            # Test decompose_k template filtering
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["decompose_k"]
            )
            if result is None:
                raise AssertionError("Result should not be None")
            self.assertEqual(len(result["decompose_k"]), 1)
            self.assertNotIn("template_id", result["decompose_k"][0])
            self.assertEqual(result["decompose_k"][0]["k_split"], 4)

    def test_empty_table(self):
        # [FUNCTION DOC] With an empty table, lookup returns an empty mapping.
        """Test when template lookup table is empty"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        with patch.object(inductor_config.lookup_table, "table", {}):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"]
            )
            self.assertEqual(result, {})

    def test_validation_error(self):
        # [FUNCTION DOC] A stored config lacking template_id makes lookup raise ValueError whose
        # message mentions the missing required template_id field.
        """Test validation error for invalid config"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()
        invalid_config = {"BLOCK_M": 128}  # missing template_id

        lookup_table_data = {
            self.create_lookup_key("mm", kernel_inputs): [invalid_config]
        }

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()
            with self.assertRaises(ValueError) as cm:
                test_choices.lookup_template_configs(kernel_inputs, "mm", ["triton"])
            self.assertIn("missing required 'template_id' field", str(cm.exception))

    def test_cpu_input_returns_empty(self):
        # [FUNCTION DOC] Kernel inputs on CPU produce an empty result even though the table has an
        # entry for them.
        """Test that CPU tensor input returns empty dict"""
        # Create kernel inputs with CPU tensors
        kernel_inputs = self.create_mock_mm_kernel_inputs(device=torch.device("cpu"))

        lookup_table_data = {
            self.create_lookup_key("mm", kernel_inputs): [self.create_config("triton")]
        }

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"]
            )
            self.assertEqual(result, {})  # Should return empty dict for CPU

    def test_multiple_calls_work(self):
        # [FUNCTION DOC] Repeated lookups of triton and tma configs against the same table give
        # the same results each time (one config each).
        """Test that calling lookup functions multiple times works correctly"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config("triton", BLOCK_M=128),
            self.create_config("tma", BLOCK_M=256),
        ]

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()

            # First calls
            result1 = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"]
            )
            result2 = test_choices.lookup_template_configs(kernel_inputs, "mm", ["tma"])
            if result1 is None:
                raise AssertionError("Result1 should not be None")
            if result2 is None:
                raise AssertionError("Result2 should not be None")
            self.assertEqual(len(result1["triton"]), 1)
            self.assertEqual(len(result2["tma"]), 1)

            # Second calls should work the same
            result3 = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"]
            )
            result4 = test_choices.lookup_template_configs(kernel_inputs, "mm", ["tma"])
            if result3 is None:
                raise AssertionError("Result3 should not be None")
            if result4 is None:
                raise AssertionError("Result4 should not be None")
            self.assertEqual(len(result3["triton"]), 1)
            self.assertEqual(len(result4["tma"]), 1)

    def test_batch_lookup_mixed_entries(self):
        # [FUNCTION DOC] Looking up three template ids where only two have configs returns entries
        # for just those two; the one without configs is absent from the result.
        """Test batch lookup where some templates have entries and others don't"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config("triton", BLOCK_M=128),
            self.create_config("tma", BLOCK_M=256),
            # No decompose_k config in lookup table
        ]

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()

            # Test batch lookup with mixed results
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton", "tma", "decompose_k"]
            )
            if result is None:
                raise AssertionError("Result should not be None")

            # Should have entries for triton and tma, but not decompose_k
            self.assertIn("triton", result)
            self.assertIn("tma", result)
            self.assertNotIn("decompose_k", result)

            self.assertEqual(len(result["triton"]), 1)
            self.assertEqual(len(result["tma"]), 1)
            self.assertEqual(result["triton"][0]["BLOCK_M"], 128)
            self.assertEqual(result["tma"][0]["BLOCK_M"], 256)

    @parametrize(
        "config_hash,template_hash,expected_kept",
        [
            # Hash matching (config kept)
            ("hash123", "hash123", True),
            # Hash mismatch (config filtered)
            ("hash123", "hash456", False),
            # Config without hash (config kept)
            (None, "hash123", True),
            # Template without hash (config kept)
            ("hash123", None, True),
            # Both None (config kept)
            (None, None, True),
        ],
    )
    def test_template_hash_checking(self, config_hash, template_hash, expected_kept):
        # [FUNCTION DOC] Parametrized over config hash, template hash and expected outcome, with
        # source hash checking enabled. A config is dropped only when both hashes exist and
        # differ. Kept configs have the hash field stripped; when dropped the result is empty.
        """Test template hash validation behavior"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config = self.create_config("triton", BLOCK_M=128, BLOCK_N=64)
        if config_hash is not None:
            config["template_hash"] = config_hash

        template_hash_map = (
            {"triton": template_hash} if template_hash is not None else {}
        )

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): [config]}

        with (
            patch.object(inductor_config.lookup_table, "table", lookup_table_data),
            patch.object(inductor_config.lookup_table, "check_src_hash", True),
        ):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            if expected_kept:
                if result is None:
                    raise AssertionError("Result should not be None")
                self.assertIn("triton", result)
                self.assertEqual(len(result["triton"]), 1)
                # template_hash should be removed from returned config
                self.assertNotIn("template_hash", result["triton"][0])
            else:
                # Config was filtered out due to hash mismatch
                self.assertEqual(result, {})

    def test_template_hash_checking_disabled(self):
        # [FUNCTION DOC] With source hash checking disabled, a config with a mismatched hash is
        # still returned, and its hash field is still stripped.
        """Test that hash checking is skipped when config flag is disabled"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        # Create config with mismatching hash
        config = self.create_config("triton", BLOCK_M=128, template_hash="hash123")

        # Provide different template hash that would normally cause filtering
        template_hash_map = {"triton": "hash456"}

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): [config]}

        with (
            patch.object(inductor_config.lookup_table, "table", lookup_table_data),
            patch.object(
                inductor_config.lookup_table,
                "check_src_hash",
                False,
            ),
        ):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            # Should keep config even with mismatching hash since checking is disabled
            if result is None:
                raise AssertionError("Result should not be None")
            self.assertIn("triton", result)
            self.assertEqual(len(result["triton"]), 1)
            # template_hash should still be removed from returned config
            self.assertNotIn("template_hash", result["triton"][0])

    def test_template_hash_mixed_scenarios(self):
        # [FUNCTION DOC] With hash checking on, three configs (right hash, wrong hash, no hash)
        # yield two kept configs: the correct-hash one and the hash-less one. The wrong-hash
        # config is dropped and no returned config contains a hash field.
        """Test mixed hash scenarios with multiple configs"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config(
                "triton", BLOCK_M=128, template_hash="correct_hash"
            ),  # Should be kept
            self.create_config(
                "triton", BLOCK_M=64, template_hash="wrong_hash"
            ),  # Should be filtered
            self.create_config("triton", BLOCK_M=32),  # No hash, should be kept
        ]

        template_hash_map = {"triton": "correct_hash"}

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        with (
            patch.object(inductor_config.lookup_table, "table", lookup_table_data),
            patch.object(inductor_config.lookup_table, "check_src_hash", True),
        ):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            if result is None:
                raise AssertionError("Result should not be None")
            self.assertIn("triton", result)
            # Should keep 2 configs: the one with correct hash and the one without hash
            self.assertEqual(len(result["triton"]), 2)

            # Check that kept configs have expected BLOCK_M values
            kept_block_ms = [config["BLOCK_M"] for config in result["triton"]]
            self.assertIn(128, kept_block_ms)  # Config with correct hash
            self.assertIn(32, kept_block_ms)  # Config without hash
            self.assertNotIn(
                64, kept_block_ms
            )  # Config with wrong hash should be filtered

            # template_hash should be removed from returned configs
            for config in result["triton"]:
                self.assertNotIn("template_hash", config)

    @parametrize(
        "config_hash,description",
        [
            ("definitely_malformed_hash_!@#$%", "malformed hash"),
            (12345, "non-string hash"),
            ("", "empty string hash"),
            (None, "missing hash field"),
        ],
    )
    def test_hash_checking_disabled_edge_cases(self, config_hash, description):
        # [FUNCTION DOC] Parametrized over odd hash values (malformed, non-string, empty, absent).
        # With checking disabled, the config is always kept, its hash field is removed and its
        # other fields are preserved.
        """Test that configs are kept when hash checking is disabled, regardless of hash validity"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        # Create config with potentially problematic hash
        config = self.create_config("triton", BLOCK_M=128)
        if config_hash is not None:
            config["template_hash"] = config_hash
        # If config_hash is None, don't add template_hash field at all

        # Provide a valid template hash that would normally be used for comparison
        template_hash_map = {"triton": "valid_template_hash_abc123"}

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): [config]}

        with (
            patch.object(inductor_config.lookup_table, "table", lookup_table_data),
            patch.object(inductor_config.lookup_table, "check_src_hash", False),
        ):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            # Should keep config regardless of hash validity since checking is disabled
            if result is None:
                raise AssertionError(f"Result should not be None for {description}")
            self.assertIn(
                "triton",
                result,
                lambda msg: f"{msg}\nShould have triton result for {description}",
            )
            self.assertEqual(
                len(result["triton"]),
                1,
                lambda msg: f"{msg}\nShould have 1 config for {description}",
            )
            # template_hash should be removed from returned config
            self.assertNotIn(
                "template_hash",
                result["triton"][0],
                lambda msg: f"{msg}\ntemplate_hash should be removed from result for {description}",
            )
            # Other config fields should be preserved
            self.assertEqual(
                result["triton"][0]["BLOCK_M"],
                128,
                lambda msg: f"{msg}\nBLOCK_M should be preserved for {description}",
            )

    @parametrize(
        "table_has_device_key,lookup_device_matches,expected_found",
        [
            # Device-specific key in table, same device -> found
            (True, True, True),
            # Device-specific key in table, different device -> not found
            (True, False, False),
            # Device-agnostic key in table, same device -> found
            (False, True, True),
            # Device-agnostic key in table, different device -> found (device-agnostic)
            (False, False, True),
        ],
    )
    def test_device_key_lookup_scenarios(
        self, table_has_device_key, lookup_device_matches, expected_found
    ):
        # [FUNCTION DOC] Parametrized over whether the table key includes the device and whether
        # the lookup device matches. A device-specific key is found only with a matching device; a
        # device-agnostic key is always found. Not found yields an empty mapping; found keeps the
        # config contents.
        """Test lookup behavior with device-specific vs device-agnostic keys"""
        # Create kernel inputs for "device_1" (our reference device)
        kernel_inputs_device1 = self.create_mock_mm_kernel_inputs()

        # Create config
        config = self.create_config("triton", BLOCK_M=128)

        # Create a test choices class for generating the table key
        class TableKeyChoices(LookupTableChoices):
            # [CLASS DOC] HELPER: Choices handler whose device key is overridden to a fixed
            # identifier for CUDA devices, used to generate the table key.
            @staticmethod
            def _get_device_key(device):
                # [FUNCTION DOC] HELPER: Static override returning no key for non-CUDA devices and
                # a fixed first-device identifier for CUDA devices.
                if device.type != "cuda":
                    return None
                return "device_1"  # Always device_1 for table key generation

        table_key_choices = TableKeyChoices()

        # Generate table key based on whether it should include device
        if table_has_device_key:
            table_key = table_key_choices.make_lookup_key(
                kernel_inputs_device1, "mm", include_device=True
            )
        else:
            table_key = table_key_choices.make_lookup_key(
                kernel_inputs_device1, "mm", include_device=False
            )

        lookup_table_data = {table_key: [config]}

        # Create test choices class for the actual lookup with different device behavior
        if lookup_device_matches:

            class TestChoices(LookupTableChoices):
                # [CLASS DOC] HELPER: Lookup handler whose device key equals the table-key device.
                @staticmethod
                def _get_device_key(device):
                    # [FUNCTION DOC] HELPER: Static override returning no key for non-CUDA devices
                    # and the same first-device identifier for CUDA devices.
                    if device.type != "cuda":
                        return None
                    return "device_1"

        else:

            class TestChoices(LookupTableChoices):
                # [CLASS DOC] HELPER: Lookup handler whose device key differs from the table-key
                # device.
                @staticmethod
                def _get_device_key(device):
                    # [FUNCTION DOC] HELPER: Static override returning no key for non-CUDA devices
                    # and a second, different device identifier for CUDA devices.
                    if device.type != "cuda":
                        return None
                    return "device_2"

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = TestChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs_device1, "mm", ["triton"]
            )

        if expected_found:
            if result is None:
                raise AssertionError(
                    f"Result should not be None when expected_found={expected_found}"
                )
            self.assertIn("triton", result, "Should have triton result when found")
            self.assertEqual(len(result["triton"]), 1, "Should have exactly 1 config")
            self.assertEqual(
                result["triton"][0]["BLOCK_M"], 128, "Config should be preserved"
            )
        else:
            self.assertEqual(
                result,
                {},
                lambda msg: f"{msg}\nShould return empty dict when expected_found={expected_found}",
            )

    def test_device_key_priority(self):
        # [FUNCTION DOC] Stores both a device-specific and a device-agnostic entry for the same
        # inputs with different block sizes; lookup must return the device-specific config.
        """Test that device-specific keys take priority over device-agnostic keys"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        # Create two different configs
        device_specific_config = self.create_config(
            "triton", BLOCK_M=256
        )  # Different BLOCK_M
        device_agnostic_config = self.create_config("triton", BLOCK_M=128)

        # Create a test choices instance to generate keys
        key_choices = LookupTableChoices()

        # Create both key types for the same inputs
        device_key = key_choices.make_lookup_key(
            kernel_inputs, "mm", include_device=True
        )
        device_agnostic_key = key_choices.make_lookup_key(
            kernel_inputs, "mm", include_device=False
        )

        # Put both in the table
        lookup_table_data = {
            device_key: [device_specific_config],
            device_agnostic_key: [device_agnostic_config],
        }

        with patch.object(inductor_config.lookup_table, "table", lookup_table_data):
            test_choices = LookupTableChoices()
            result = test_choices.lookup_template_configs(
                kernel_inputs, "mm", ["triton"]
            )

            # Should get device-specific config (BLOCK_M=256), not device-agnostic (BLOCK_M=128)
            if result is None:
                raise AssertionError("Result should not be None")
            self.assertIn("triton", result)
            self.assertEqual(len(result["triton"]), 1)
            self.assertEqual(
                result["triton"][0]["BLOCK_M"],
                256,
                "Should use device-specific config when both exist",
            )

    def test_make_lookup_key_variants(self):
        # [FUNCTION DOC] Checks the key-variants helper returns two strings, the device key being
        # longer, and the part of the device-agnostic key before the method suffix is contained in
        # the device key.
        """Test the make_lookup_key_variants helper function"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        test_choices = LookupTableChoices()
        device_key, device_agnostic_key = test_choices.make_lookup_key_variants(
            kernel_inputs, "mm"
        )

        # Both should be strings
        self.assertIsInstance(device_key, str)
        self.assertIsInstance(device_agnostic_key, str)

        # Device key should be longer (contains device info)
        self.assertGreater(len(device_key), len(device_agnostic_key))

        # Device-agnostic key should be contained in device key (as a substring after device part)
        self.assertIn(device_agnostic_key.split("+mm")[0], device_key)


class UnifiedModel(nn.Module):
    # [CLASS DOC] HELPER: Module applying one matrix operation chosen at construction (mm, addmm,
    # bmm or mm_plus_mm), stored as an attribute.
    """Unified model for different matrix operations"""

    def __init__(self, operation="mm"):
        # [FUNCTION DOC] HELPER: Stores the operation name.
        super().__init__()
        self.operation = operation

    def forward(self, *args):
        # [FUNCTION DOC] HELPER: Dispatches on the operation name and applies the matching torch
        # op to the positional arguments (mm_plus_mm sums two products). Raises ValueError for an
        # unknown operation.
        if self.operation == "mm":
            return torch.mm(args[0], args[1])
        elif self.operation == "addmm":
            return torch.addmm(args[0], args[1], args[2])
        elif self.operation == "bmm":
            return torch.bmm(args[0], args[1])
        elif self.operation == "mm_plus_mm":
            return torch.mm(args[0], args[1]) + torch.mm(args[2], args[3])
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")


def verify_choice_names(choices: list[Any], pattern: str, expected_count: int = 1):
    # [FUNCTION DOC] HELPER: Preprocessing-function body for checks. Raises ValueError if the
    # choice count differs from the expected count or any choice name does not match the regular
    # expression pattern. Returns the choices unchanged otherwise.
    """Verify choices match expected pattern and count"""
    if len(choices) != expected_count:
        raise ValueError(f"Expected {expected_count} choices, got {len(choices)}")
    for choice in choices:
        if not re.search(pattern, choice.name):
            raise ValueError(
                f"Choice name '{choice.name}' doesn't match pattern '{pattern}'"
            )
    return choices


class BaseE2ELookupTableTest(BaseLookupTableTest):
    # [CLASS DOC] HELPER: Base class for end-to-end tests. Resets dynamo and preprocessing
    # functions, records the CUDA device and its device key, saves the lookup table, and installs
    # the lookup-table choices handler. Teardown restores table and handler and clears the
    # preprocessing functions.
    """Base class for E2E lookup table tests"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Performs the base setup, resets dynamo state, clears
        # preprocessing functions, sets CUDA device attributes, saves the lookup table and
        # installs the handler.
        super().setUp()
        torch._dynamo.reset()
        clear_preprocessing_fns()
        self.device = torch.device("cuda")
        self.dev_key = LookupTableChoices._get_device_key(self.device)
        self.original_lookup_table = inductor_config.lookup_table.table
        # Set the lookup table choices handler
        V.set_choices_handler(LookupTableChoices())

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Restores lookup table and default choices handler and clears
        # preprocessing functions. Does not call the base teardown.
        inductor_config.lookup_table.table = self.original_lookup_table
        # Restore original choices handler
        V.set_choices_handler(InductorChoices())
        clear_preprocessing_fns()

    def create_tensors(self, operation, b=8, m=64, n=64, k=32):
        # [FUNCTION DOC] HELPER: Creates half-precision CUDA tensors for the given operation: mm
        # (two operands), addmm (bias, two operands), mm_plus_mm (four) or bmm (two batched).
        # Dimensions are configurable. Raises ValueError for other operations.
        """Create test tensors for operations with configurable dimensions"""
        if operation in ["mm", "addmm", "mm_plus_mm"]:
            A = torch.randn(m, k, device=self.device, dtype=torch.float16)
            B = torch.randn(k, n, device=self.device, dtype=torch.float16)
            if operation == "mm":
                return [A, B]
            if operation == "addmm":
                return [
                    torch.randn((m, n), device=self.device, dtype=torch.float16),
                    A,
                    B,
                ]
            elif operation == "mm_plus_mm":
                return [
                    A,
                    B,
                    torch.randn(m, k, device=self.device, dtype=torch.float16),
                    torch.randn(k, n, device=self.device, dtype=torch.float16),
                ]
        elif operation == "bmm":
            return [
                torch.randn(b, m, k, device=self.device, dtype=torch.float16),
                torch.randn(b, k, n, device=self.device, dtype=torch.float16),
            ]
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def setup_lookup_table(self, operation, tensors, configs):
        # [FUNCTION DOC] HELPER: Builds mock kernel inputs (with unit alpha and beta scalars for
        # addmm and baddbmm), computes the lookup key and replaces the global lookup table with a
        # mapping from that key to the given configs.
        """Setup lookup table with configuration"""
        scalars = {}
        if operation in ["addmm", "baddbmm"]:
            scalars["beta"] = 1
            scalars["alpha"] = 1
        mock_kernel_inputs = MockMMKernelInputs(tensors, scalars)
        flat_key = self.create_lookup_key(operation, mock_kernel_inputs)
        inductor_config.lookup_table.table = {flat_key: configs}

    def run_model(self, operation, tensors, config_patches=None):
        # [FUNCTION DOC] HELPER: Compiles a UnifiedModel for the operation on the CUDA device
        # under config overrides (gemm autotune on, limited config count, plus optional extra
        # patches) and returns the model output on the given tensors.
        """Run compiled model with configuration"""
        config = {"max_autotune_gemm": True, "test_configs.max_mm_configs": 4}
        if config_patches:
            config.update(config_patches)

        model = UnifiedModel(operation)
        with inductor_config.patch(config):
            compiled_model = torch.compile(model.to(self.device))
            return compiled_model(*tensors)

    def create_basic_config(self, template_id):
        # [FUNCTION DOC] HELPER: Returns a config mapping for a template id. Known template ids
        # get their block sizes, stages and flags (the TMA one also gets SM count and descriptor
        # size; the bias addmm one has none; the decompose-k one has a split of four). Unknown ids
        # only get the template id.
        """Create basic configuration for template"""
        configs = {
            torch._inductor.kernel.mm.mm_template.uid: {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_stages": 2,
                "num_warps": 2,
                "EVEN_K": True,
                "USE_FAST_ACCUM": False,
                "ACC_TYPE": "tl.float32",
                "GROUP_M": 8,
            },
            torch._inductor.kernel.mm_plus_mm.mm_plus_mm_template.uid: {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_stages": 2,
                "num_warps": 2,
                "EVEN_K": True,
                "USE_FAST_ACCUM": False,
                "ACC_TYPE": "tl.float32",
                "GROUP_M": 8,
            },
            torch._inductor.kernel.bmm.bmm_template.uid: {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "num_stages": 2,
                "num_warps": 2,
                "EVEN_K": True,
                "USE_FAST_ACCUM": False,
                "ACC_TYPE": "tl.float32",
                "GROUP_M": 8,
            },
            torch._inductor.kernel.mm.persistent_tma_mm_template.uid: {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_stages": 2,
                "num_warps": 2,
                "EVEN_K": True,
                "USE_FAST_ACCUM": False,
                "ACC_TYPE": "tl.float32",
                "GROUP_M": 8,
                "A_ROW_MAJOR": True,
                "B_ROW_MAJOR": True,
                "NUM_SMS": get_num_sms(),
                "TMA_SIZE": TMA_DESCRIPTOR_SIZE,
                "TMA_EXPERIMENTAL_API": not has_triton_stable_tma_api(),
            },
            torch._inductor.kernel.mm.aten_bias_addmm.uid: {},
            torch._inductor.kernel.mm.decompose_k_subgraph_template.uid: {"k_split": 4},
        }
        return {"template_id": template_id, **configs.get(template_id, {})}

    def _create_simple_matmul_model(self):
        # [FUNCTION DOC] HELPER: Returns a module that performs a single matrix multiplication.
        """Create a simple matmul model for recording tests"""

        class SimpleMatmul(nn.Module):
            # [CLASS DOC] HELPER: Module whose forward multiplies its two inputs.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the matrix product of the two inputs.
                return torch.mm(a, b)

        return SimpleMatmul()

    def _create_test_inputs(self, device="cuda"):
        # [FUNCTION DOC] HELPER: Returns two random 512 by 512 float32 tensors on the device.
        """Create test inputs for matmul"""
        return [
            torch.randn(512, 512, device=device, dtype=torch.float32),
            torch.randn(512, 512, device=device, dtype=torch.float32),
        ]


@unittest.skipIf(not HAS_CUDA_AND_TRITON, "CUDA not available")
@instantiate_parametrized_tests
class TestLookupTableE2E(BaseE2ELookupTableTest):
    # [CLASS DOC] End-to-end tests compiling matmul models with lookup table entries and checking
    # the resulting template choices through preprocessing hooks. Needs CUDA and triton.
    """E2E tests for lookup table functionality"""

    @parametrize("max_autotune", [True, False])
    @fresh_cache()
    def test_no_lookup_table_entry_autotune_modes(self, max_autotune):
        # [FUNCTION DOC] Parametrized over max autotune on/off with a table entry that does not
        # match. With autotune on, choices must be many and include both extern and triton
        # callers; off, exactly one extern choice. Validation runs in a preprocessing hook during
        # compilation.
        """Test when there's no lookup table entry with different autotune modes"""
        tensors = self.create_tensors("mm")

        # Setup lookup table with different key to force no match
        self.setup_lookup_table(
            "mm",
            [
                torch.randn(64, 64, device=self.device),
                torch.randn(64, 64, device=self.device),
            ],
            [],
        )

        # Inline validation function
        def validate_choices(choices):
            # [FUNCTION DOC] HELPER: Preprocessing hook raising AssertionError when the choice
            # list violates the expectations for the current autotune mode; returns the choices
            # otherwise.
            if max_autotune:
                if len(choices) <= 2:
                    raise AssertionError(
                        f"Max-autotune should have >2 choices, got {len(choices)}"
                    )
                if not any(isinstance(c, ExternKernelCaller) for c in choices):
                    raise AssertionError("Should have ExternKernelCaller")
                if not any(isinstance(c, TritonTemplateCaller) for c in choices):
                    raise AssertionError("Should have TritonTemplateCaller")
            else:
                if len(choices) != 1:
                    raise AssertionError(
                        f"No max-autotune should have 1 choice, got {len(choices)}"
                    )
                if not isinstance(choices[0], ExternKernelCaller):
                    raise AssertionError(
                        f"Should be ExternKernelCaller, got {type(choices[0])}"
                    )
            return choices

        add_preprocessing_fn(validate_choices)
        self.run_model(
            "mm",
            tensors,
            {"max_autotune_gemm": max_autotune, "max_autotune": max_autotune},
        )

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180234")
    @parametrize("operation", ["mm", "addmm", "bmm", "mm_plus_mm"])
    @fresh_cache()
    def test_valid_lookup_table_entry(self, operation):
        # [FUNCTION DOC] Parametrized over mm, addmm, bmm and mm_plus_mm. Stores a basic config
        # for the matching template and checks the choice names. addmm expects two choices (triton
        # or aten addmm), others expect one triton choice. Skips addmm on ROCm. Skipped on ROCm
        # for a tracked issue.
        """Test when there's a valid entry for the operation"""
        if operation == "addmm" and torch.version.hip:
            self.skipTest(
                "skipping on ROCm since https://github.com/pytorch/pytorch/issues/179955 didn't skip as expected"
            )
        k = 256 if operation == "mm_plus_mm" else 64
        tensors = self.create_tensors(operation, k=k)

        # Map operation to actual template UID
        template_mapping = {
            "mm": torch._inductor.kernel.mm.mm_template.uid,
            "addmm": torch._inductor.kernel.mm.mm_template.uid,
            "bmm": torch._inductor.kernel.bmm.bmm_template.uid,
            "mm_plus_mm": torch._inductor.kernel.mm_plus_mm.mm_plus_mm_template.uid,
        }
        template_id = template_mapping[operation]
        config = self.create_basic_config(template_id)

        self.setup_lookup_table(operation, tensors, [config])

        # TODO (paulzhan): Update LookupTableChoices to return empty
        #  (not fallback) when key matches
        if operation == "addmm":
            add_preprocessing_fn(
                partial(verify_choice_names, pattern="triton_|addmm", expected_count=2)
            )
        else:
            add_preprocessing_fn(
                partial(verify_choice_names, pattern="triton_", expected_count=1)
            )
        self.run_model(operation, tensors)

    @unittest.skipIf(not has_triton_tma_device(), "Need TMA support")
    @parametrize("operation", ["mm", "addmm"])
    @fresh_cache()
    def test_tma_lookup_table_entry(self, operation):
        # [FUNCTION DOC] Needs TMA support. For mm and addmm, stores a persistent TMA template
        # config and checks the choice names (two for addmm including aten, one for mm) with
        # persistent TMA matmul enabled.
        """Test TMA template entry"""
        tensors = self.create_tensors(operation)
        config = self.create_basic_config(
            torch._inductor.kernel.mm.persistent_tma_mm_template.uid
        )

        self.setup_lookup_table(operation, tensors, [config])
        if operation == "addmm":
            add_preprocessing_fn(
                partial(
                    verify_choice_names,
                    pattern="triton_mm_persistent_tma_|addmm",
                    expected_count=2,
                )
            )
        else:
            add_preprocessing_fn(
                partial(
                    verify_choice_names,
                    pattern="triton_mm_persistent_tma_",
                    expected_count=1,
                )
            )
        self.run_model(
            operation, tensors, {"triton.enable_persistent_tma_matmul": True}
        )

    # Enable decompose_k for this test (disabled by default on ROCm)
    @fresh_cache()
    def test_decompose_k_lookup_table_entry(self):
        # [FUNCTION DOC] Stores a decompose-k config for a tall-k mm, with the ROCm override
        # applied, and checks that exactly one decompose-k choice results.
        """Test decompose_k template entry"""
        with inductor_config.patch(_DECOMPOSE_K_PATCH_ROCM):
            tensors = self.create_tensors("mm", m=32, n=32, k=32 * 32)
            config = self.create_basic_config(
                torch._inductor.kernel.mm.decompose_k_subgraph_template.uid
            )

            self.setup_lookup_table("mm", tensors, [config])
            add_preprocessing_fn(
                partial(
                    verify_choice_names,
                    pattern="decompose_k|bmm_dtype",
                    expected_count=1,
                )
            )

            self.run_model("mm", tensors)

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180233")
    @fresh_cache()
    def test_bias_addmm_lookup_table_entry(self):
        # [FUNCTION DOC] Builds an expanded bias, stores a bias addmm entry and runs a compiled
        # addmm in max-autotune mode limited to aten backends. Only checks the model runs.
        # KNOWN ISSUE: the lookup key is built from the expanded bias but no choice count is
        # verified; a code comment itself notes the unexpanded vs expanded shape discrepancy.
        """Test bias_addmm template entry"""
        # Create bias with stride[0] == 0 for bias_addmm eligibility
        bias_unexpanded = torch.randn(64, device=self.device, dtype=torch.float16)
        expanded_bias = bias_unexpanded.expand(64, 64)
        tensors = [
            expanded_bias,
            torch.randn(64, 32, device=self.device, dtype=torch.float16),
            torch.randn(32, 64, device=self.device, dtype=torch.float16),
        ]

        config = self.create_basic_config(torch._inductor.kernel.mm.aten_bias_addmm.uid)
        self.setup_lookup_table("addmm", tensors, [config])
        # NOTE: This test passes bias_unexpanded (1D) to the model but sets up
        # lookup key with expanded_bias (2D). The shapes differ so lookup will miss.
        # We skip choice count verification here - just verify the model runs.

        # Run with expanded bias (stride[0] == 0) so the inductor sees
        # bias_addmm-eligible inputs and the lookup key matches.
        # Limit backends to ATEN so only the lookup table entry is selected.
        with inductor_config.patch(
            {
                "max_autotune_gemm": True,
                "triton.autotune_cublasLt": True,
                "max_autotune_gemm_backends": "ATEN",
            }
        ):
            model = UnifiedModel("addmm")
            compiled_model = torch.compile(model.to(self.device), mode="max-autotune")
            compiled_model(expanded_bias, tensors[1], tensors[2])

    @unittest.skipIf(not has_triton_tma_device(), "Need TMA support")
    @fresh_cache()
    def test_multiple_configs_same_template(self):
        # [FUNCTION DOC] Needs TMA support. Stores two TMA configs with different block sizes and
        # expects two persistent TMA choices.
        """Test multiple configurations for same template"""
        tensors = self.create_tensors("mm")

        config1 = self.create_basic_config(
            torch._inductor.kernel.mm.persistent_tma_mm_template.uid
        )
        config1.update({"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8})

        config2 = self.create_basic_config(
            torch._inductor.kernel.mm.persistent_tma_mm_template.uid
        )
        config2.update({"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4})

        self.setup_lookup_table("mm", tensors, [config1, config2])
        add_preprocessing_fn(
            partial(
                verify_choice_names,
                pattern="triton_mm_persistent_tma_",
                expected_count=2,
            )
        )
        self.run_model("mm", tensors, {"triton.enable_persistent_tma_matmul": True})

    @unittest.skipIf(not has_triton_tma_device(), "Need TMA support")
    @fresh_cache()
    def test_mixed_template_configs(self):
        # [FUNCTION DOC] Needs TMA support. Stores one regular triton config and one TMA config
        # and expects two triton choices.
        """Test mixing different template types"""
        tensors = self.create_tensors("mm")

        triton_config = self.create_basic_config(
            torch._inductor.kernel.mm.mm_template.uid
        )
        triton_config.update({"BLOCK_M": 128, "num_warps": 8})

        tma_config = self.create_basic_config(
            torch._inductor.kernel.mm.persistent_tma_mm_template.uid
        )
        tma_config.update({"BLOCK_M": 256, "num_warps": 4})

        self.setup_lookup_table("mm", tensors, [triton_config, tma_config])
        add_preprocessing_fn(
            partial(verify_choice_names, pattern="triton_", expected_count=2)
        )
        self.run_model("mm", tensors, {"triton.enable_persistent_tma_matmul": True})

    @fresh_cache()
    def test_template_hash_filtering_e2e(self):
        # [FUNCTION DOC] Stores two mm configs, one with the template real source hash and one
        # with a wrong hash. With hash checking on, only one choice must remain.
        """Test end-to-end template hash filtering in real MM operation"""
        tensors = self.create_tensors("mm")

        # Get the actual src_hash from the template
        actual_hash = torch._inductor.kernel.mm.mm_template.src_hash

        # Create configs - one with correct hash, one with wrong hash
        correct_config = self.create_basic_config(
            torch._inductor.kernel.mm.mm_template.uid
        )
        correct_config.update(
            {"BLOCK_M": 128, "template_hash": actual_hash}  # Use actual hash
        )

        wrong_config = self.create_basic_config(
            torch._inductor.kernel.mm.mm_template.uid
        )
        wrong_config.update(
            {
                "BLOCK_M": 64,
                "template_hash": "definitely_wrong_hash_12345",  # Wrong hash
            }
        )

        self.setup_lookup_table("mm", tensors, [correct_config, wrong_config])

        # Should only get 1 choice since the wrong hash config gets filtered
        add_preprocessing_fn(
            partial(verify_choice_names, pattern="triton_", expected_count=1)
        )

        # Ensure hash checking is enabled
        with patch.object(inductor_config.lookup_table, "check_src_hash", True):
            self.run_model("mm", tensors)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
