# [MODULE DOC] Defines the QuantType enum (dynamic, static, QAT, weight-only) used to
# classify quantization modes, plus helpers to convert a QuantType to and from its
# lowercase string name. Exports only QuantType. No import-time side effects.
import enum


__all__ = [
    "QuantType",
]


# Quantization type (dynamic quantization, static quantization).
# Should match the c++ enum in quantization_type.h
class QuantType(enum.IntEnum):
    # [CLASS DOC] Integer enum of quantization modes: DYNAMIC=0, STATIC=1, QAT=2,
    # WEIGHT_ONLY=3. Values must stay in sync with the C++ enum in quantization_type.h.
    DYNAMIC = 0
    STATIC = 1
    QAT = 2
    WEIGHT_ONLY = 3


_quant_type_to_str = {
    QuantType.STATIC: "static",
    QuantType.DYNAMIC: "dynamic",
    QuantType.QAT: "qat",
    QuantType.WEIGHT_ONLY: "weight_only",
}


# TODO: make this private
def _get_quant_type_to_str(quant_type: QuantType) -> str:
    # [FUNCTION DOC] HELPER: Returns the lowercase string name ("static", "dynamic", "qat",
    # "weight_only") for a QuantType; raises a lookup error for an unmapped value.
    return _quant_type_to_str[quant_type]


def _quant_type_from_str(name: str) -> QuantType:
    # [FUNCTION DOC] HELPER: Inverse of the string mapping; scans the mapping for the given
    # name and returns the matching QuantType. Raises ValueError if the name is unknown.
    for quant_type, s in _quant_type_to_str.items():
        if name == s:
            return quant_type
    raise ValueError(f"Unknown QuantType name '{name}'")
