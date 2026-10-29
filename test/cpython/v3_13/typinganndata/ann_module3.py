# [MODULE DOC] HELPER: fixture of annotation syntax that is accepted by the compiler but fails at
# runtime in specific ways (undefined names). Nothing is executed at import time.
"""
Correct syntax for variable annotation that should fail at runtime
in a certain manner. More examples are in test_grammar and test_parser.
"""

def f_bad_ann():
    # [FUNCTION DOC] HELPER: assigns into the annotations mapping with a non-string key; when
    # called it raises NameError because a function scope has no such local mapping.
    __annotations__[1] = 2

class C_OK:
    # [CLASS DOC] HELPER: initializer annotates an attribute with an undefined name; accepted at
    # runtime because annotations of attribute targets are never evaluated.
    def __init__(self, x: int) -> None:
        # [FUNCTION DOC] HELPER: stores x on self; the undefined name in the annotation is not
        # evaluated, so this works.
        self.x: no_such_name = x  # This one is OK as proposed by Guido

class D_bad_ann:
    # [CLASS DOC] HELPER: initializer whose annotated assignment targets an undefined name.
    def __init__(self, x: int) -> None:
        # [FUNCTION DOC] HELPER: raises NameError when called, because the target name sfel is
        # undefined and the annotated assignment still evaluates its target.
        sfel.y: int = 0

def g_bad_ann():
    # [FUNCTION DOC] HELPER: raises NameError when called, because no_such_name is undefined.
    no_such_name.attr: int = 0
