# [MODULE DOC] HELPER: fixture with valid variable-annotation syntax: annotated module names, an
# annotated attribute assignment on a function result and on an instance, a module annotations
# mapping reset to empty, a no_type_check class and a ClassVar attribute. Import-time side
# effects: creates instance c and assigns CV.var an instance of CV.
"""
Some correct syntax for variable annotation here.
More examples are in test_grammar and test_parser.
"""

from typing import no_type_check, ClassVar

i: int = 1
j: int
x: float = i/10

def f():
    # [FUNCTION DOC] HELPER: defines a local empty class C and returns a new instance of it.
    class C: ...
    return C()

f().new_attr: object = object()

class C:
    # [CLASS DOC] HELPER: stores an int argument x on the instance; annotation is on the
    # initializer parameter only.
    def __init__(self, x: int) -> None:
        # [FUNCTION DOC] HELPER: saves x as an instance attribute.
        self.x = x

c = C(5)
c.new_attr: int = 10

__annotations__ = {}


@no_type_check
class NTC:
    # [CLASS DOC] HELPER: class decorated with no_type_check so typing helpers ignore its hints.
    def meth(self, param: complex) -> None:
        # [FUNCTION DOC] HELPER: placeholder method that does nothing.
        ...

class CV:
    # [CLASS DOC] HELPER: declares a ClassVar annotation var naming the class itself (forward
    # reference); the value is assigned after the class body.
    var: ClassVar['CV']

CV.var = CV()
