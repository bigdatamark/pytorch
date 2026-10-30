# [PACKAGE DOC] Fixture package that imports its submodules via explicit dynamic import calls
# with relative levels, to test dependency detection for such imports. Import-time side effect:
# imports subpackage_1, subpackage_0.subsubpackage_0 and subpackage_2. Exports a result string
# and the PackageBObject wrapper class.
__import__("subpackage_1", globals(), fromlist=["PackageBSubpackage1Object_0"], level=1)
__import__("subpackage_0.subsubpackage_0", globals(), fromlist=[""], level=1)
__import__("subpackage_2", globals=globals(), locals=locals(), fromlist=["*"], level=1)

result = "package_b"


class PackageBObject:
    # [CLASS DOC] Minimal wrapper with a single slot holding an arbitrary object; return_result
    # reports the package-level constant.
    __slots__ = ["obj"]

    def __init__(self, obj):
        # [FUNCTION DOC] HELPER: Stores the given object in the single slot attribute.
        self.obj = obj

    def return_result(self):
        # [FUNCTION DOC] Returns the package-level result string, looked up in module globals at
        # call time.
        return result
