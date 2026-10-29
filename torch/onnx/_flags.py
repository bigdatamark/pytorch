# [MODULE DOC] Experimental environment-variable feature flags for the ONNX exporter.
# Import-time side effects: reads the process environment and, for each flag set to the
# value 1, logs a warning (or error if deprecated). Defines ENABLE_DRAFT_EXPORT (default
# False) and PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS (default True) as module booleans.
"""Internal feature flags for torch.onnx.

NOTE: These flags are experimental only. Any flag here can be removed at any
time without notice.
"""

import logging
import os


logger = logging.getLogger(__name__)


def _load_boolean_flag(
    name: str,
    *,
    this_will: str,
    deprecated: bool = False,
    default: bool = False,
) -> bool:
    # [FUNCTION DOC] HELPER: Read a boolean flag from an environment variable.
    # The flag is on only when the variable equals the string 1; an unset variable yields
    # default instead. A variable set to anything else is off, even if default is True.
    # Logs an error for deprecated flags or a warning describing this_will when enabled.
    """Load a boolean flag from environment variable.

    Args:
        name: The name of the environment variable.
        this_will: A string that describes what this flag will do.
        deprecated: Whether this flag is deprecated.
        default: The default value if envvar not defined.
    """
    undefined = os.getenv(name) is None
    state = os.getenv(name) == "1"
    if state:
        if deprecated:
            logger.error(
                "Experimental flag %s is deprecated. Please remove it from your environment.",
                name,
            )
        else:
            logger.warning(
                "Experimental flag %s is enabled. This will %s.", name, this_will
            )
    if undefined:
        state = default
    return state


ENABLE_DRAFT_EXPORT: bool = _load_boolean_flag(
    "TORCH_ONNX_ENABLE_DRAFT_EXPORT",
    this_will="enable torch.export.draft_export as a strategy for capturing models",
    default=False,
)
PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS: bool = _load_boolean_flag(
    "TORCH_ONNX_PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS",
    this_will="set prefer_deferred_runtime_asserts_over_guards when calling torch.export",
    default=True,
)
