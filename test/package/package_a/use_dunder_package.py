# [MODULE DOC] Fixture that defines a function reporting whether the module was loaded by the
# package importer. At import time it checks whether the module namespace contains the torch
# package marker name, which the importer injects, and defines one of two variants.
if "__torch_package__" in dir():

    def is_from_package():
        # [FUNCTION DOC] HELPER: Variant defined when the package marker is present;
        # returns True.
        return True

else:

    def is_from_package():
        # [FUNCTION DOC] HELPER: Variant defined when the package marker is absent;
        # returns False.
        return False
