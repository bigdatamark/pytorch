# [PACKAGE DOC] Fixture sub-subpackage of package_b. Import-time side effect: dynamically
# imports subpackage_1 using a relative level of three, i.e. three packages up. Exports a result
# string and an empty marker class.
__import__("subpackage_1", globals(), locals(), ["PackageBSubpackage1Object_0"], 3)

result = "subsubpackage_0"


class PackageBSubsubpackage0Object_0:
    # [CLASS DOC] HELPER: Empty marker class used as a picklable object defined in this package.
    pass
