# [MODULE DOC] Fixture that tries to import the torch_package_importer name, which exists only
# when the module is loaded from inside a package; an import failure is silently ignored.
try:
    import torch_package_importer  # noqa: F401
except ImportError:
    pass
