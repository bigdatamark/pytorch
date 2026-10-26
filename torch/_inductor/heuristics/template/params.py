# [MODULE DOC] Parameter objects wrapping a single template config: an abstract interface and a simple dict-backed implementation. No import-time side effects.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class KernelTemplateParams(ABC):
    # [CLASS DOC] Abstract interface for a kernel template config: convertible to kwargs, to a serializable dict, and constructible from a dict.
    """Abstract base class for kernel template parameters."""

    @abstractmethod
    def to_kwargs(self) -> dict[str, Any]:
        # [FUNCTION DOC] Abstract: return a kwargs dict for the template choice creation.
        """Convert params to kwargs dict for template.choice_or_none()"""

    @abstractmethod
    def to_serializeable_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Abstract: return a serializable dict for storage or caching.
        """Convert params to serializable dict for storage/caching"""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> KernelTemplateParams:
        # [FUNCTION DOC] Abstract classmethod: build a params instance from a dict.
        """Create params instance from dict"""


class DictKernelTemplateParams(KernelTemplateParams):
    # [CLASS DOC] Compatibility params implementation that stores a plain kwargs dict in the attribute kwargs.
    """Simple implementation that wraps a kwargs dict"""

    # NOTE: this is a compatibility layer, until every template
    # has time to define their own params class, with meaningful
    # defaults etc.

    def __init__(self, kwargs: dict[str, Any]):
        # [FUNCTION DOC] Store the given kwargs dict (not copied) as the kwargs attribute.
        self.kwargs = kwargs

    def to_kwargs(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return a shallow copy of the kwargs dict.
        return self.kwargs.copy()

    def to_serializeable_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return a shallow copy of the kwargs dict (same as to_kwargs).
        return self.kwargs.copy()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DictKernelTemplateParams:
        # [FUNCTION DOC] Classmethod: build an instance wrapping the given dict (not copied).
        return cls(data)
