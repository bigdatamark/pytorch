# Owner(s): ["oncall: export"]
# [MODULE DOC] Runs the whole TestExport suite a second time with torch.export.export replaced
# by a mocked version that round-trips every ExportedProgram through the C++ serializer.
# At import time it builds a dynamic subclass named with the CppSerdes prefix via
# testing.make_test_cls_with_mocked_export, registers it in this module's globals (required
# for test discovery) and then deletes the loop variable. Tests listed as expected failures
# via the _expected_failure_cpp_serdes property are marked xfail. Running the file directly
# starts the dynamo test runner.


import torch
from torch._export.serde.serialize import deserialize, serialize


try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export


test_classes = {}


def mocked_cpp_serdes_export(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Drop-in replacement for export that also exercises C++ serde.
    # Exports with the real export, then tries to serialize the program in Python; if that
    # raises any exception the unmodified exported program is returned. Otherwise the payload is
    # deserialized by the C++ binding, re-serialized to JSON by the C++ binding, the JSON text is
    # stored back (encoded) in the payload, and the Python deserializer rebuilds and returns the
    # program. Forwards all arguments to export unchanged.
    ep = export(*args, **kwargs)
    try:
        payload = serialize(ep)
    except Exception:
        return ep
    cpp_ep = torch._C._export.deserialize_exported_program(payload.exported_program)
    loaded_json = torch._C._export.serialize_exported_program(cpp_ep)
    payload.exported_program = loaded_json.encode()
    loaded_ep = deserialize(payload)
    return loaded_ep


def make_dynamic_cls(cls):
    # [FUNCTION DOC] HELPER: Builds and registers a test class derived from the given test class
    # whose export calls go through mocked_cpp_serdes_export. The new class gets the CppSerdes
    # name prefix and the _cpp_serdes suffix, uses the _expected_failure_cpp_serdes property for
    # xfails, is stored in test_classes and in module globals (so the runner finds it), and has
    # its module name set to this module. Returns nothing.
    cls_prefix = "CppSerdes"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        "_cpp_serdes",
        mocked_cpp_serdes_export,
        xfail_prop="_expected_failure_cpp_serdes",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__


tests = [
    test_export.TestExport,
]
for test in tests:
    make_dynamic_cls(test)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
