# [MODULE DOC] Fixture submodule of package_b that uses dynamic import calls. Import-time side
# effect: imports math and xml.sax.xmlreader. Also defines a result string, an empty marker
# class and a function that imports an arbitrary named module.
__import__("math", fromlist=[])
__import__("xml.sax.xmlreader")

result = "subpackage_2"


class PackageBSubpackage2Object_0:
    # [CLASS DOC] HELPER: Empty marker class used as a picklable object defined in this module.
    pass


def dynamic_import_test(name: str):
    # [FUNCTION DOC] HELPER: Imports the module with the given name at call time through the
    # dynamic import builtin and returns None. An unknown name raises an import error. Used to
    # check that packaging cannot see dependencies named only by runtime strings.
    __import__(name)
