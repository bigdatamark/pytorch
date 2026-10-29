# [MODULE DOC] Deprecated compatibility shim: re-exports CustomClassBase and
# CustomClassBaseMeta from torch._custom_class_base under the old names
# OpaqueBase and OpaqueBaseMeta.
# Import-time side effect: logs a deprecation warning on the module logger.
import logging


log = logging.getLogger(__name__)

log.warning("torch._opaque_base is deprecated, use torch._custom_class_base instead")

from torch._custom_class_base import (  # noqa: F401
    CustomClassBase as OpaqueBase,
    CustomClassBaseMeta as OpaqueBaseMeta,
)
