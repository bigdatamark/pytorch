# [MODULE DOC]
# Helper callables referenced by name from pickle archives written by
# ScriptModule.save(), so Python's unpickler can load the data. Mostly identity
# functions kept for backward compatibility. No import-time side effects.
# These functions are referenced from the pickle archives produced by
# ScriptModule.save()


# These (`build_*`) functions used to be used by `pickler.cpp` to specify
# the type of the list for certain special types, but now all lists get
# a type attached and restored via `restore_type_tag` below. The legacy
# functions should stick around for backwards-compatibility.


def build_intlist(data: list[int]) -> list[int]:
    # [FUNCTION DOC] HELPER: Legacy unpickling hook; returns the given list of
    # ints unchanged.
    return data


def build_tensorlist(data: list[object]) -> list[object]:
    # [FUNCTION DOC] HELPER: Legacy unpickling hook; returns the given list of
    # tensors unchanged.
    return data


def build_doublelist(data: list[float]) -> list[float]:
    # [FUNCTION DOC] HELPER: Legacy unpickling hook; returns the given list of
    # floats unchanged.
    return data


def build_boollist(data: list[bool]) -> list[bool]:
    # [FUNCTION DOC] HELPER: Legacy unpickling hook; returns the given list of
    # bools unchanged.
    return data


def build_tensor_from_id(data: int | object) -> int | None:
    # [FUNCTION DOC] HELPER: Legacy unpickling hook for a tensor referenced by
    # id. Returns the integer id itself if data is an int, otherwise None; no
    # tensor is actually reconstructed.
    if isinstance(data, int):
        # just the id, can't really do anything
        return data
    return None


def restore_type_tag(value: object, type_str: str) -> object:
    # [FUNCTION DOC] HELPER: Unpickling hook that receives a container value and
    # its static type string; the type string is ignored in Python and the value
    # is returned unchanged.
    # The type_ptr is used by the jit unpickler to restore the full static type
    # to container types like list when they are re-loaded, but this doesn't
    # matter for Python, so just return the plain value
    return value
