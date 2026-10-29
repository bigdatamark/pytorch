# Owner(s): ["module: sparse"]

# [MODULE DOC]
# Tests for the fully-qualified-name (fqn) helpers in torch.ao.pruning.sparsifier.utils:
# module_to_fqn, fqn_to_module and get_arg_info_from_tensor_fqn, run over a list of
# quantization test models. Import-time side effect: configures root logging at INFO
# level. Running the file directly raises an error pointing at test/test_ao_sparsity.py.


import logging

import torch
from torch.ao.pruning.sparsifier.utils import (
    fqn_to_module,
    get_arg_info_from_tensor_fqn,
    module_to_fqn,
)
from torch.testing._internal.common_quantization import (
    ConvBnReLUModel,
    ConvModel,
    FunctionalLinear,
    LinearAddModel,
    ManualEmbeddingBagLinear,
    SingleLayerLinearModel,
    TwoLayerLinearModel,
)
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

model_list = [
    ConvModel,
    SingleLayerLinearModel,
    TwoLayerLinearModel,
    LinearAddModel,
    ConvBnReLUModel,
    ManualEmbeddingBagLinear,
    FunctionalLinear,
]


class TestSparsityUtilFunctions(TestCase):
    # [CLASS DOC] Test case exercising the sparsity fqn utilities against every model
    # class in the module-level model_list; stateless.
    def test_module_to_fqn(self):
        # [FUNCTION DOC] For each model and each of its submodules (plus the model),
        # computes the fqn and asserts get_submodule on that fqn returns the same module.
        """
        Tests that module_to_fqn works as expected when compared to known good
        module.get_submodule(fqn) function
        """
        for model_class in model_list:
            model = model_class()
            list_of_modules = [m for _, m in model.named_modules()] + [model]
            for module in list_of_modules:
                fqn = module_to_fqn(model, module)
                check_module = model.get_submodule(fqn)
                self.assertEqual(module, check_module)

    def test_module_to_fqn_fail(self):
        # [FUNCTION DOC] Asserts module_to_fqn returns None for a module that is not
        # part of the model.
        """
        Tests that module_to_fqn returns None when an fqn that doesn't
        correspond to a path to a node/tensor is given
        """
        for model_class in model_list:
            model = model_class()
            fqn = module_to_fqn(model, torch.nn.Linear(3, 3))
            self.assertEqual(fqn, None)

    def test_module_to_fqn_root(self):
        # [FUNCTION DOC] Asserts module_to_fqn returns the empty string when the target
        # module is the model itself.
        """
        Tests that module_to_fqn returns '' when model and target module are the same
        """
        for model_class in model_list:
            model = model_class()
            fqn = module_to_fqn(model, model)
            self.assertEqual(fqn, "")

    def test_fqn_to_module(self):
        # [FUNCTION DOC] Round trip check: module to fqn to module returns the original
        # module for every submodule and the root of each model.
        """
        Tests that fqn_to_module operates as inverse
        of module_to_fqn
        """
        for model_class in model_list:
            model = model_class()
            list_of_modules = [m for _, m in model.named_modules()] + [model]
            for module in list_of_modules:
                fqn = module_to_fqn(model, module)
                check_module = fqn_to_module(model, fqn)
                self.assertEqual(module, check_module)

    def test_fqn_to_module_fail(self):
        # [FUNCTION DOC] Asserts fqn_to_module returns None for a nonexistent fqn.
        """
        Tests that fqn_to_module returns None when it tries to
        find an fqn of a module outside the model
        """
        for model_class in model_list:
            model = model_class()
            fqn = "foo.bar.baz"
            check_module = fqn_to_module(model, fqn)
            self.assertEqual(check_module, None)

    def test_fqn_to_module_for_tensors(self):
        # [FUNCTION DOC] For every directly-owned parameter of every module, builds the
        # tensor fqn (module fqn, a dot unless root, tensor name) and asserts
        # fqn_to_module resolves it to that parameter.
        """
        Tests that fqn_to_module works for tensors, actually all parameters
        of the model. This is tested by identifying a module with a tensor,
        and generating the tensor_fqn using module_to_fqn on the module +
        the name of the tensor.
        """
        for model_class in model_list:
            model = model_class()
            list_of_modules = [m for _, m in model.named_modules()] + [model]
            for module in list_of_modules:
                module_fqn = module_to_fqn(model, module)
                for tensor_name, tensor in module.named_parameters(recurse=False):
                    tensor_fqn = (  # string manip to handle tensors on root
                        module_fqn + ("." if module_fqn != "" else "") + tensor_name
                    )
                    check_tensor = fqn_to_module(model, tensor_fqn)
                    self.assertEqual(tensor, check_tensor)

    def test_get_arg_info_from_tensor_fqn(self):
        # [FUNCTION DOC] For every parameter, asserts the returned mapping holds the
        # owning module, its module_fqn, the tensor_name and the full tensor_fqn.
        """
        Tests that get_arg_info_from_tensor_fqn works for all parameters of the model.
        Generates a tensor_fqn in the same way as test_fqn_to_module_for_tensors and
        then compares with known (parent) module and tensor_name as well as module_fqn
        from module_to_fqn.
        """
        for model_class in model_list:
            model = model_class()
            list_of_modules = [m for _, m in model.named_modules()] + [model]
            for module in list_of_modules:
                module_fqn = module_to_fqn(model, module)
                for tensor_name, _ in module.named_parameters(recurse=False):
                    tensor_fqn = (
                        module_fqn + ("." if module_fqn != "" else "") + tensor_name
                    )
                    arg_info = get_arg_info_from_tensor_fqn(model, tensor_fqn)
                    self.assertEqual(arg_info["module"], module)
                    self.assertEqual(arg_info["module_fqn"], module_fqn)
                    self.assertEqual(arg_info["tensor_name"], tensor_name)
                    self.assertEqual(arg_info["tensor_fqn"], tensor_fqn)

    def test_get_arg_info_from_tensor_fqn_fail(self):
        # [FUNCTION DOC] For a bogus fqn, asserts the module entry is None while the
        # string entries (module_fqn, tensor_name, tensor_fqn) are still split correctly.
        """
        Tests that get_arg_info_from_tensor_fqn works as expected for invalid tensor_fqn
        inputs. The string outputs still work but the output module is expected to be None.
        """
        for model_class in model_list:
            model = model_class()
            tensor_fqn = "foo.bar.baz"
            arg_info = get_arg_info_from_tensor_fqn(model, tensor_fqn)
            self.assertEqual(arg_info["module"], None)
            self.assertEqual(arg_info["module_fqn"], "foo.bar")
            self.assertEqual(arg_info["tensor_name"], "baz")
            self.assertEqual(arg_info["tensor_fqn"], "foo.bar.baz")


if __name__ == "__main__":
    raise_on_run_directly("test/test_ao_sparsity.py")
