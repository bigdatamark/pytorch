# [MODULE DOC] HELPER: fixture for no_type_check on nested classes and functions: an outer class
# with an inner class holding an int attribute, and a function with int annotations.
# Test `@no_type_check`,
# see https://bugs.python.org/issue46571

class NoTypeCheck_Outer:
    # [CLASS DOC] HELPER: container whose nested class Inner is the subject of the check.
    class Inner:
        # [CLASS DOC] HELPER: nested class with a single int-annotated attribute x.
        x: int


def NoTypeCheck_function(arg: int) -> int:
    # [FUNCTION DOC] HELPER: stub with int parameter and return annotation; returns None.
    ...
