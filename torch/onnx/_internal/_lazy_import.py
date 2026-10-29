# [MODULE DOC] Lazy proxies for optional ONNX dependencies (onnx, onnx_ir, onnxscript and
# its torch 2.11 framework APIs). Exposes module-like objects that import the real module
# on first attribute access; under type checking the real imports are used instead.
# Import time creates the proxies only, without importing the dependencies.
"""Utility to lazily import modules."""

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING


class _LazyModule:
    # [CLASS DOC] HELPER: Proxy that imports a named module on first attribute access.
    # State: the module name and the cached module (None until loaded). Attribute lookups are
    # forwarded to the imported module; import errors surface at that first access.
    """Lazily import a module."""

    def __init__(self, module_name: str) -> None:
        # [FUNCTION DOC] HELPER: Remember the module name; nothing is imported yet.
        self._name = module_name
        self._module: Any = None

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Return a readable representation naming the lazy module.
        return f"<lazy module '{self._name}'>"

    def __getattr__(self, attr: str) -> object:
        # [FUNCTION DOC] HELPER: Import the module if needed and return the requested attribute.
        # Raises whatever the import or attribute lookup raises.
        if self._module is None:
            self._module = importlib.import_module(".", self._name)
        return getattr(self._module, attr)


# Import the following modules during type checking to enable code intelligence features,
# such as auto-completion in tools like pylance, even when these modules are not explicitly
# imported in user code.
# NOTE: Add additional used imports here.
if TYPE_CHECKING:
    import onnx
    import onnx_ir  # type: ignore[import-untyped, import-not-found]
    import onnxscript
    import onnxscript._framework_apis.torch_2_11 as onnxscript_apis

else:
    onnx = _LazyModule("onnx")
    onnx_ir = _LazyModule("onnx_ir")
    onnxscript = _LazyModule("onnxscript")
    onnxscript_apis = _LazyModule("onnxscript._framework_apis.torch_2_11")
