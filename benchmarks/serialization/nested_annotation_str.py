# [MODULE DOC] Script measuring how long it takes to compute the annotation string of deeply
# nested jit dictionary types. It builds dictionary types nested 1, 3, 5 and 10 levels deep
# (each level maps strings to a pair of the previous level), times the annotation string
# property with the benchmark timer, and prints the nesting level, the first 70 characters of
# the string, and the timing for each. Everything runs at import time; a module-level cache
# avoids rebuilding types.
import torch
import torch.utils.benchmark as benchmark


MEMO = {}


def create_nested_dict_type(layers):
    # [FUNCTION DOC] HELPER: Returns a jit type for the given nesting depth: the string type at
    # depth zero, otherwise a dictionary from string to a two-element tuple whose elements are
    # both the type one level shallower. Results for depths above zero are cached in the
    # module-level mapping so shared subtypes are built once.
    if layers == 0:
        return torch._C.StringType.get()
    if layers not in MEMO:
        less_nested = create_nested_dict_type(layers - 1)
        result = torch._C.DictType(
            torch._C.StringType.get(), torch._C.TupleType([less_nested, less_nested])
        )
        MEMO[layers] = result
    return MEMO[layers]


nesting_levels = (1, 3, 5, 10)
types = (reasonable, medium, big, huge) = [
    create_nested_dict_type(x) for x in nesting_levels
]

timers = [
    benchmark.Timer(stmt="x.annotation_str", globals={"x": nested_type})
    for nested_type in types
]

for nesting_level, typ, timer in zip(nesting_levels, types, timers):
    print("Nesting level:", nesting_level)
    print("output:", typ.annotation_str[:70])
    print(timer.blocked_autorange())
