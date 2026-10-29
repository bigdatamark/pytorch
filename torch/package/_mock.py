# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Support for mocked modules in torch.package. Defines the list of magic
# method names and the MockedObject placeholder class. At import time, every name in the list
# is installed on MockedObject as a method that raises NotImplementedError when used, so a
# mocked-out object fails loudly on any operation (mutates MockedObject at import).
_magic_methods = [
    "__subclasscheck__",
    "__hex__",
    "__rmul__",
    "__float__",
    "__idiv__",
    "__setattr__",
    "__div__",
    "__invert__",
    "__nonzero__",
    "__rshift__",
    "__eq__",
    "__pos__",
    "__round__",
    "__rand__",
    "__or__",
    "__complex__",
    "__divmod__",
    "__len__",
    "__reversed__",
    "__copy__",
    "__reduce__",
    "__deepcopy__",
    "__rdivmod__",
    "__rrshift__",
    "__ifloordiv__",
    "__hash__",
    "__iand__",
    "__xor__",
    "__isub__",
    "__oct__",
    "__ceil__",
    "__imod__",
    "__add__",
    "__truediv__",
    "__unicode__",
    "__le__",
    "__delitem__",
    "__sizeof__",
    "__sub__",
    "__ne__",
    "__pow__",
    "__bytes__",
    "__mul__",
    "__itruediv__",
    "__bool__",
    "__iter__",
    "__abs__",
    "__gt__",
    "__iadd__",
    "__enter__",
    "__floordiv__",
    "__call__",
    "__neg__",
    "__and__",
    "__ixor__",
    "__getitem__",
    "__exit__",
    "__cmp__",
    "__getstate__",
    "__index__",
    "__contains__",
    "__floor__",
    "__lt__",
    "__getattr__",
    "__mod__",
    "__trunc__",
    "__delattr__",
    "__instancecheck__",
    "__setitem__",
    "__ipow__",
    "__ilshift__",
    "__long__",
    "__irshift__",
    "__imul__",
    "__lshift__",
    "__dir__",
    "__ge__",
    "__int__",
    "__ior__",
]


class MockedObject:
    # [CLASS DOC] HELPER: Placeholder handed out for attributes of mocked modules. Stores the
    # fully qualified name of the mocked thing in the instance dict entry for the name
    # attribute. Constructing it is only allowed internally (with the suppress flag); any
    # other construction, e.g. during unpickling, raises NotImplementedError. All magic
    # methods are replaced (see install_method) with ones that raise NotImplementedError.
    _name: str

    def __new__(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Allocates an instance. Raises NotImplementedError unless the
        # keyword argument _suppress_err is truthy; otherwise defers to the normal object
        # allocation and returns the new instance.
        # KNOWN ISSUE: the error message reads the name from the class, but the name is only
        # set on instances, so a raise here would hit an AttributeError instead.
        # _suppress_err is set by us in the mocked module impl, so that we can
        # construct instances of MockedObject to hand out to people looking up
        # module attributes.

        # Any other attempt to construct a MockedObject instance (say, in the
        # unpickling process) should give an error.
        if not kwargs.get("_suppress_err"):
            raise NotImplementedError(
                f"Object '{cls._name}' was mocked out during packaging "
                f"but it is being used in '__new__'. If this error is "
                "happening during 'load_pickle', please ensure that your "
                "pickled object doesn't contain any mocked objects."
            )
        # Otherwise, this is just a regular object creation
        # (e.g. `x = MockedObject("foo")`), so pass it through normally.
        return super().__new__(cls)

    def __init__(self, name: str, _suppress_err: bool):
        # [FUNCTION DOC] HELPER: Records the mocked object's name directly in the instance
        # dict (bypassing the overridden attribute setter). The suppress flag is unused here.
        self.__dict__["_name"] = name

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a string showing the mocked object's name.
        return f"MockedObject({self._name})"


def install_method(method_name):
    # [FUNCTION DOC] HELPER: Installs on MockedObject a method with the given name that always
    # raises NotImplementedError naming the mocked object. Mutates the MockedObject class.
    def _not_implemented(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Always raises NotImplementedError saying the mocked object
        # was used in the enclosing method name.
        raise NotImplementedError(
            f"Object '{self._name}' was mocked out during packaging but it is being used in {method_name}"
        )

    setattr(MockedObject, method_name, _not_implemented)


for method_name in _magic_methods:
    install_method(method_name)
