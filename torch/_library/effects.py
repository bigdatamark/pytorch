# [MODULE DOC] Effect tracking for custom operators: defines EffectType, a list of
# opaque script classes that need no ordering, and EffectHolder which stores the
# effect registered for an operator (defaulting to ORDERED when an argument is a
# script object). No import-time side effects beyond definitions.
from enum import Enum

import torch


class EffectType(Enum):
    # [CLASS DOC] Enumeration of operator side-effect kinds; currently only ORDERED,
    # meaning calls must keep their relative order in traced graphs.
    ORDERED = "Ordered"


from torch._library.utils import RegistrationHandle


# These classes do not have side effects as they just store quantization
# params, so we don't need to mark them as ordered
skip_classes = (
    "__torch__.torch.classes.quantized.Conv2dPackedParamsBase",
    "__torch__.torch.classes.quantized.Conv3dPackedParamsBase",
    "__torch__.torch.classes.quantized.EmbeddingPackedParamsBase",
    "__torch__.torch.classes.quantized.LinearPackedParamsBase",
    "__torch__.torch.classes.xnnpack.Conv2dOpContext",
    "__torch__.torch.classes.xnnpack.LinearOpContext",
    "__torch__.torch.classes.xnnpack.TransposeConv2dOpContext",
)


class EffectHolder:
    # [CLASS DOC] Per-operator holder of an optional EffectType. Attributes: qualname
    # (operator qualified name) and _effect (current effect or None). The effect is read
    # via a read-only property and changed only through register(), which returns a
    # handle that restores the default effect.
    """A holder where one can register an effect impl to."""

    def __init__(self, qualname: str):
        # [FUNCTION DOC] Stores qualname and initializes the effect to its default.
        self.qualname: str = qualname
        self._set_default_effect()

    def _set_default_effect(self) -> None:
        # [FUNCTION DOC] HELPER: Resets _effect to None, then sets it to ORDERED if the
        # operator (non higher_order namespace, schema still present) has any argument of
        # a script class type not in skip_classes. Raises AssertionError if the name has
        # more than one '.' separating overload. Reads global operator registry.
        self._effect: EffectType | None = None

        # If the op contains a ScriptObject input, we want to mark it as having effects
        namespace, opname = torch._library.utils.parse_namespace(self.qualname)
        split = opname.split(".")
        if len(split) > 1:
            if len(split) != 2:
                raise AssertionError(
                    f"Tried to split {opname} based on '.' but found more than 1 '.'"
                )
            opname, overload = split
        else:
            overload = ""

        if namespace == "higher_order":
            return

        opname = f"{namespace}::{opname}"
        if torch._C._get_operation_overload(opname, overload) is not None:
            # Since we call this when destroying the library, sometimes the
            # schema will be gone already at that time.
            schema = torch._C._get_schema(opname, overload)
            for arg in schema.arguments:
                if isinstance(arg.type, torch.ClassType):
                    type_str = arg.type.str()  # pyrefly: ignore[missing-attribute]
                    if type_str in skip_classes:
                        continue
                    self._effect = EffectType.ORDERED
                    return

    @property
    def effect(self) -> EffectType | None:
        # [FUNCTION DOC] Property getter: returns the current effect or None.
        return self._effect

    @effect.setter
    def effect(self, _):
        # [FUNCTION DOC] Property setter: always raises RuntimeError; use register instead.
        raise RuntimeError("Unable to directly set kernel.")

    def register(self, effect: EffectType | None) -> RegistrationHandle:
        # [FUNCTION DOC] Sets the effect (may be None) and returns a RegistrationHandle
        # whose deregistration recomputes the default effect. Mutates holder state.
        """Register an effect

        Returns a RegistrationHandle that one can use to de-register this
        effect.
        """
        self._effect = effect

        def deregister_effect():
            # [FUNCTION DOC] HELPER: Restores the default effect on the holder.
            self._set_default_effect()

        handle = RegistrationHandle(deregister_effect)
        return handle
