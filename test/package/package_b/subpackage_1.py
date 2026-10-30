# [MODULE DOC] Fixture submodule of package_b with a result string and a one-slot wrapper class.
result = "subpackage_1"


class PackageBSubpackage1Object_0:
    # [CLASS DOC] HELPER: Wrapper with a single slot holding an arbitrary object; return_result
    # reports the module-level constant.
    __slots__ = ["obj"]

    def __init__(self, obj):
        # [FUNCTION DOC] HELPER: Stores the given object in the single slot attribute.
        self.obj = obj

    def return_result(self):
        # [FUNCTION DOC] HELPER: Returns the module-level result string.
        return result
