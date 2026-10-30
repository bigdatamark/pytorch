# [MODULE DOC] Fixture submodule of package_a: a string constant naming itself, an empty marker
# class and a small addition function.
result = "package_a.subpackage"


class PackageASubpackageObject:
    # [CLASS DOC] HELPER: Empty marker class used as a picklable object defined in
    # the subpackage.
    pass


def leaf_function(a, b):
    # [FUNCTION DOC] HELPER: Returns the sum of the two arguments.
    return a + b
