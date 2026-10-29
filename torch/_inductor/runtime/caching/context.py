# [MODULE DOC] Context providers for Inductor runtime caching.
# Defines runtime and compile context collectors (config, determinism, matmul precision,
# torch/triton/runtime versions, accelerator properties), an isolation schema selecting
# which forms to include, and helpers to build a context dict and a 32-char hash key.
# No import-time side effects beyond imports.
"""Context management for PyTorch Inductor runtime caching.

This module provides context classes for collecting configuration and environment
information used in caching decisions for PyTorch's Inductor runtime.
"""

import json
from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import Sequence
from functools import cache
from hashlib import sha256
from typing_extensions import override, TypedDict

import torch


class _Context(ABC):
    # [CLASS DOC] Abstract base for context providers; subclasses expose static
    # collector methods and list their names via forms_of_context.
    """Abstract base class for context providers.

    Context providers collect specific configuration and environment information
    that affects compilation and runtime behavior.
    """

    @staticmethod
    @abstractmethod
    def forms_of_context() -> Sequence[str]:
        # [FUNCTION DOC] Abstract static method; returns the names of the context forms (collector
        # method names) the subclass provides.
        """Return a sequence of context form names provided by this context class.

        Returns:
            A sequence of strings representing the available context forms.
        """


class _RuntimeContext(_Context):
    # [CLASS DOC] Provider of settings that affect runtime behavior only: inductor configs,
    # torch determinism flags and CUDA matmul precision flags. All collectors are static and
    # read live global state each call (no caching).
    """Context provider for runtime configuration and environment settings.

    Collects configuration settings that affect runtime behavior but not
    compilation, such as Inductor configs, determinism settings, and CUDA
    matmul precision configurations.
    """

    @override
    @staticmethod
    def forms_of_context() -> Sequence[str]:
        # [FUNCTION DOC] Returns a tuple of the three runtime form names: inductor_configs,
        # torch_determinism_configs, cuda_matmul_precision_configs.
        """Return the runtime context forms provided by this class.

        Returns:
            A sequence containing the available runtime context forms:
            - "inductor_configs": PyTorch Inductor configuration settings
            - "torch_determinism_configs": Deterministic algorithm settings
            - "cuda_matmul_precision_configs": CUDA matrix multiplication precision settings
        """
        return (
            "inductor_configs",
            "torch_determinism_configs",
            "cuda_matmul_precision_configs",
        )

    @staticmethod
    def inductor_configs() -> dict[str, object]:
        # [FUNCTION DOC] Returns the portable Inductor config as a mapping, including private configs.
        # Imports the inductor config module lazily.
        """Get portable Inductor configuration settings.

        Returns:
            A dictionary containing Inductor configuration settings,
            including private configs.
        """
        from torch._inductor import config

        return config.save_config_portable(ignore_private_configs=False)

    @staticmethod
    def torch_determinism_configs() -> dict[str, object]:
        # [FUNCTION DOC] Returns a mapping of the deterministic-algorithms enabled flag, the warn-only flag,
        # and the fill-uninitialized-memory setting (None if the attribute is missing).
        """Get PyTorch deterministic algorithm configuration settings.

        Returns:
            A dictionary containing deterministic algorithm settings:
            - Whether deterministic algorithms are enabled
            - Whether deterministic algorithm warnings are enabled
            - Fill uninitialized memory setting
        """
        return {
            "torch.are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "torch.is_deterministic_algorithms_warn_only_enabled": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "torch.utils.deterministic.fill_uninitialized_memory": getattr(
                torch.utils.deterministic, "fill_uninitialized_memory", None
            ),
        }

    @staticmethod
    def cuda_matmul_precision_configs() -> dict[str, object]:
        # [FUNCTION DOC] Returns a mapping with the CUDA matmul fp32 precision setting and the fp16/bf16
        # reduced-precision-reduction allowance flags.
        """Get CUDA matrix multiplication precision configuration settings.

        Returns:
            A dictionary containing CUDA matmul precision settings:
            - FP32 precision setting
            - FP16 reduced precision reduction allowance
            - BF16 reduced precision reduction allowance
        """
        return {
            "torch.backends.cuda.matmul.fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        }


class _CompileContext(_Context):
    # [CLASS DOC] Provider of settings that affect compilation: torch version hash, triton version
    # hash, runtime kind and version, and accelerator properties. Collectors are static and
    # memoized with an unbounded cache, so values are computed once per process.
    """Context provider for compilation-related configuration and environment settings.

    Collects information that affects compilation behavior, such as PyTorch and Triton
    versions, runtime environment, and accelerator properties.
    """

    @override
    @staticmethod
    def forms_of_context() -> Sequence[str]:
        # [FUNCTION DOC] Returns a tuple of the five compile form names (torch_version_hash,
        # triton_version_hash, runtime, runtime_version, accelerator_properties).
        """Return the compile context forms provided by this class.

        Returns:
            A sequence containing the available compile context forms:
            - "torch_version_hash": PyTorch version hash
            - "triton_version_hash": Triton version hash (if available)
            - "runtime": Runtime type (CUDA/HIP/None)
            - "runtime_version": Runtime version string
            - "accelerator_properties": GPU/accelerator properties
        """
        return (
            "torch_version_hash",
            "triton_version_hash",
            "runtime",
            "runtime_version",
            "accelerator_properties",
        )

    @cache
    @staticmethod
    def torch_version_hash() -> str:
        # [FUNCTION DOC] Returns the base64-encoded torch key bytes (from codecache) as a string. Memoized.
        """Get base64-encoded PyTorch version hash.

        Returns:
            A base64-encoded string representing the PyTorch version hash.
        """
        from torch._inductor.codecache import torch_key

        return b64encode(torch_key()).decode()

    @cache
    @staticmethod
    def triton_version_hash() -> str | None:
        # [FUNCTION DOC] Returns the triton key if triton is available, otherwise None. Memoized.
        """Get Triton version key if Triton is available.

        Returns:
            Triton version key if Triton is available, None otherwise.
        """
        from torch._inductor.runtime.triton_compat import HAS_TRITON, triton_key

        return triton_key() if HAS_TRITON else None

    @cache
    @staticmethod
    def runtime() -> str | None:
        # [FUNCTION DOC] Returns "CUDA" if torch was built with CUDA, else "HIP" if built with HIP, else None. Memoized.
        """Determine the runtime type based on available backends.

        Returns:
            "CUDA" if CUDA is available, "HIP" if HIP is available, None otherwise.
        """
        return "CUDA" if torch.version.cuda else "HIP" if torch.version.hip else None

    @cache
    @staticmethod
    def runtime_version() -> str | None:
        # [FUNCTION DOC] Returns the CUDA or HIP version string matching the detected runtime, or None
        # when no runtime is detected. Memoized.
        """Get the version string for the detected runtime.

        Returns:
            Version string for the current runtime (CUDA or HIP), or None if
            no supported runtime is detected.
        """
        return {
            "CUDA": torch.version.cuda,
            "HIP": torch.version.hip,
            "None": None,
        }.get(_CompileContext.runtime() or "None")

    @cache
    @staticmethod
    def accelerator_properties() -> str | None:
        # [FUNCTION DOC] Returns the textual representation of the current CUDA device properties if a
        # runtime is built in and CUDA is available at runtime, otherwise None. Memoized, so it
        # reflects the current device at first call only.
        """Get string representation of CUDA device properties.

        Returns:
            String representation of CUDA device properties if a runtime is
            available, None otherwise.
        """
        return (
            repr(torch.cuda.get_device_properties())
            if _CompileContext.runtime() and torch.cuda.is_available()
            else None
        )


class SelectedRuntimeContext(TypedDict):
    # [CLASS DOC] Typed mapping of boolean flags selecting which runtime context forms to include.
    inductor_configs: bool
    torch_determinism_configs: bool
    cuda_matmul_precision_configs: bool


class SelectedCompileContext(TypedDict):
    # [CLASS DOC] Typed mapping of boolean flags selecting which compile context forms to include.
    torch_version_hash: bool
    triton_version_hash: bool
    runtime: bool
    runtime_version: bool
    accelerator_properties: bool


class IsolationSchema(TypedDict):
    # [CLASS DOC] Typed mapping with runtime_context and compile_context entries; each is True (all),
    # False (none) or a per-form selection mapping.
    """Schema for specifying which context forms to include in cache isolation.

    Attributes:
        runtime_context: Either True (include all runtime context), False (exclude all),
                        or a SelectedRuntimeContext dict specifying which forms to include.
        compile_context: Either True (include all compile context), False (exclude all),
                        or a SelectedCompileContext dict specifying which forms to include.
    """

    runtime_context: SelectedRuntimeContext | bool
    compile_context: SelectedCompileContext | bool


_DEFAULT_ISOLATION_SCHEMA: IsolationSchema = IsolationSchema(
    runtime_context=True, compile_context=True
)


def _collect_runtime_context(
    selection: SelectedRuntimeContext | bool,
) -> dict[str, object] | None:
    # [FUNCTION DOC] HELPER: Collects the selected runtime context forms into a mapping of form name to
    # value. Selection True includes all; a mapping includes forms whose flag is set. Returns
    # None if nothing is selected.
    """Collect runtime context based on selection.

    Args:
        selection: True to include all, False to exclude all, or a dict
                  specifying which forms to include.

    Returns:
        Dictionary of selected context data, or None if excluded.
    """
    return {
        form: getattr(_RuntimeContext, form)()
        for form in _RuntimeContext.forms_of_context()
        if selection is True or (selection and selection.get(form, False))
    } or None


def _collect_compile_context(
    selection: SelectedCompileContext | bool,
) -> dict[str, object] | None:
    # [FUNCTION DOC] HELPER: Same as the runtime collector but for compile context forms. Returns a mapping
    # of form name to value, or None if nothing is selected.
    """Collect compile context based on selection.

    Args:
        selection: True to include all, False to exclude all, or a dict
                  specifying which forms to include.

    Returns:
        Dictionary of selected context data, or None if excluded.
    """
    return {
        form: getattr(_CompileContext, form)()
        for form in _CompileContext.forms_of_context()
        if selection is True or (selection and selection.get(form, False))
    } or None


def _isolation_context(
    ischema: IsolationSchema = _DEFAULT_ISOLATION_SCHEMA,
) -> dict[str, object]:
    # [FUNCTION DOC] HELPER: Builds a mapping with keys runtime_context and compile_context, each holding the
    # collected form values or None, according to the isolation schema (default: everything).
    """Generate context data based on the isolation schema.

    Args:
        ischema: Schema specifying which context forms to include.
                Defaults to including all runtime and compile context.

    Returns:
        A dictionary containing the selected context data with keys
        "runtime_context" and "compile_context", where each value is
        either None (if excluded) or a dict of context form data.
    """
    return {
        "runtime_context": _collect_runtime_context(ischema["runtime_context"]),
        "compile_context": _collect_compile_context(ischema["compile_context"]),
    }


def _isolation_key(ischema: IsolationSchema = _DEFAULT_ISOLATION_SCHEMA) -> str:
    # [FUNCTION DOC] HELPER: Returns the first 32 hex characters of the SHA-256 of the key-sorted JSON dump of
    # the isolation context. Raises a TypeError if a context value is not JSON serializable.
    """Generate a unique key for the given isolation schema.

    Args:
        ischema: Schema specifying which context forms to include.
                Defaults to including all runtime and compile context.

    Returns:
        A 32-character hexadecimal string that uniquely identifies
        the context specified by the isolation schema.
    """
    return sha256(
        json.dumps(_isolation_context(ischema), sort_keys=True).encode()
    ).hexdigest()[:32]
