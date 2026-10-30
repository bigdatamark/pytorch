# [PACKAGE DOC] Fixture package exporting a result string and the PackageCObject wrapper class.
result = "package_c"


class PackageCObject:
    # [CLASS DOC] Minimal wrapper with a single slot holding an arbitrary object; return_result
    # reports the package-level constant.
    __slots__ = ["obj"]

    def __init__(self, obj):
        # [FUNCTION DOC] HELPER: Stores the given object in the single slot attribute.
        self.obj = obj

    def return_result(self):
        # [FUNCTION DOC] HELPER: Returns the package-level result string.
        return result
