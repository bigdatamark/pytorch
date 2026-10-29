# [MODULE DOC] Defines ScriptObjectMeta, the metadata record attached to export graph nodes that represent
# TorchScript ScriptObjects (custom class instances). Exports ScriptObjectMeta. No import-time side effects.
from dataclasses import dataclass


__all__ = ["ScriptObjectMeta"]


@dataclass
class ScriptObjectMeta:
    # [CLASS DOC] Dataclass with two string fields: constant_name, the key into the exported program constants
    # table used to look up the real ScriptObject, and class_fqn, the fully qualified name of its custom class.
    """
    Metadata which is stored on nodes representing ScriptObjects.
    """

    # Key into constants table to retrieve the real ScriptObject.
    constant_name: str

    class_fqn: str
