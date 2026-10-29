# Owner(s): ["module: dynamo"]
# [MODULE DOC] Re-runs a fixed list of existing Dynamo test classes (ctx manager, functions, misc,
# repros, nn modules, export, subgraphs, higher-order ops, aot autograd fallback, sdpa) with
# dynamic shapes enabled. Import-time side effects: builds one derived class per source class
# with patched configs, registers each in this module's globals so the runner finds it, wraps
# three very slow export retracibility tests with slowTest, and stores classes in test_classes.
# Imports the sibling test modules relatively, falling back to plain imports when run as a script.
import warnings

from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches
from torch.fx.experimental import _config as fx_config
from torch.testing._internal.common_utils import slowTest, TEST_Z3


try:
    from . import (
        test_aot_autograd,
        test_ctx_manager,
        test_export,
        test_functions,
        test_higher_order_ops,
        test_misc,
        test_modules,
        test_repros,
        test_sdpa,
        test_subgraphs,
    )
except ImportError:
    import test_aot_autograd
    import test_ctx_manager
    import test_export
    import test_functions
    import test_higher_order_ops
    import test_misc

    import test_modules
    import test_repros
    import test_sdpa
    import test_subgraphs


test_classes = {}


def make_dynamic_cls(cls):
    # [FUNCTION DOC] Builds a dynamic-shapes variant of the given test class via
    # make_test_cls_with_patches, named with a DynamicShapes prefix and _dynamic_shapes suffix.
    # Patches: assume_static_by_default off, specialize_int off, specialize_float on (to keep
    # op counts stable), and fx translation_validation (only when Z3 is available),
    # shape-env event recording and version-key validation on. Tests listed in the class's
    # _expected_failure_dynamic set are marked as expected failures. Registers the new class in
    # test_classes and in module globals, sets its __module__ to this module, and returns it.
    suffix = "_dynamic_shapes"

    cls_prefix = "DynamicShapes"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "assume_static_by_default", False),
        (config, "specialize_int", False),
        # When we unspecialize float, we wobble tests by changing
        # the op count since previously we would just specialize and constant
        # fold floats into the graph, whereas when we unspecialize we will have
        # ops for item, add, and all other tensorified operations. Since these
        # tests really aren't testing that, we purposely specialize floats here.
        (config, "specialize_float", True),
        (fx_config, "translation_validation", TEST_Z3),
        (fx_config, "check_shape_env_recorded_events", True),
        (fx_config, "validate_shape_env_version_key", True),
        xfail_prop="_expected_failure_dynamic",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_ctx_manager.CtxManagerTests,
    test_functions.FunctionTests,
    test_misc.MiscTests,
    test_repros.ReproTests,
    test_modules.NNModuleTests,
    test_export.ExportTests,
    test_subgraphs.SubGraphTests,
    test_higher_order_ops.HigherOrderOpTests,
    test_higher_order_ops.FuncTorchHigherOrderOpTests,
    test_aot_autograd.AotAutogradFallbackTests,
    test_sdpa.TestSDPA,
]
for test in tests:
    make_dynamic_cls(test)
del test

# Test takes too long ~700s as of 414a1fd29f04d06e41b7f895368dd1f83a4be29d
DynamicShapesExportTests.test_retracibility_dynamic_shapes = slowTest(  # noqa: F821
    DynamicShapesExportTests.test_retracibility_dynamic_shapes  # noqa: F821
)
# Also take more than 30m as of 15cc9f2e7e7b2b175f24755925dc38d4d430905d
DynamicShapesExportTests.test_retracibility_dict_container_inp_out_dynamic_shapes = slowTest(  # noqa: F821
    DynamicShapesExportTests.test_retracibility_dict_container_inp_out_dynamic_shapes  # noqa: F821
)
DynamicShapesExportTests.test_retracibility_nested_list_out_dynamic_shapes = slowTest(  # noqa: F821
    DynamicShapesExportTests.test_retracibility_nested_list_out_dynamic_shapes  # noqa: F821
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not TEST_Z3:
        warnings.warn(
            "translation validation is off. "
            "Testing with translation validation requires Z3."
        )

    run_tests()
