# [PACKAGE DOC] Fake fixture package for torch.package tests. Exports a string constant naming
# the package and the PackageAObject wrapper class; sibling modules in this directory provide
# further fixtures (modules, scripted classes, tracers).
result = "package_a"


class PackageAObject:
    # [CLASS DOC] Minimal wrapper with a single slot holding an arbitrary object. Used as a
    # picklable fixture object defined in package_a; return_result reports the
    # package-level constant.
    __slots__ = ["obj"]

    def __init__(self, obj):
        # [FUNCTION DOC] HELPER: Stores the given object in the single slot attribute.
        self.obj = obj

    def return_result(self):
        # [FUNCTION DOC] Returns the package-level result string (the package name), looked up
        # in the module globals at call time.
        return result
