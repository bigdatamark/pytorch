# Owner(s): ["module: inductor"]
# [MODULE DOC] Shared helpers for AOTInductor tests: a wrapper module for plain callables, a
# utility class with legacy (shared library) and package-based compile, load and run helpers, and
# three check functions meant to be attached to test classes as methods (compare compiled against
# eager, same for several input sets, count a string in generated C++). No import-time side
# effects other than imports; the type-checking-only import is skipped at runtime. Running the
# file directly invokes the test runner.

import copy
import os
import shutil
import tempfile
import types
from typing import Any, TYPE_CHECKING

import torch
import torch._export
import torch._inductor
import torch.export._trace
import torch.fx._pytree as fx_pytree
from torch._dynamo.testing import same
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_FBCODE, run_tests
from torch.testing._internal.inductor_utils import clone_preserve_strides_offset
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from torch._C._aoti import AOTIModelContainerRunner


class WrapperModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module that wraps any callable so plain functions can be exported. Holds
    # the callable as the attribute model and forwards all positional and keyword arguments to it.
    def __init__(self, model):
        # [FUNCTION DOC] HELPER: Stores the wrapped callable.
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the wrapped callable with the given arguments and returns
        # its result.
        return self.model(*args, **kwargs)


class AOTIRunnerUtil:
    # [CLASS DOC] HELPER: Namespace of static methods (no state) that compile models with
    # AOTInductor and load or run them. Legacy methods produce and load a shared library path; the
    # non-legacy ones go through the export, package and load flow.
    @staticmethod
    def legacy_compile(
        model,
        example_inputs,
        options=None,
        dynamic_shapes=None,
        disable_constraint_solver=False,
    ):
        # [FUNCTION DOC] HELPER: Exports a model to a graph module and AOT compiles it to a shared
        # library, returning its path. Wraps non-module callables first. Uses pre-dispatch export
        # when that inductor option is set, otherwise the older export-to-IR path with the new
        # tracer config patched on and fully qualified names not restored; the constraint solver
        # can be disabled by argument. In FB code it adds a thrift extern node serializer to the
        # options (creating them if absent). Compilation runs without gradients.
        if not isinstance(model, torch.nn.Module):
            model = WrapperModule(model)
        # The exact API is subject to change
        if torch._inductor.config.is_predispatch:
            ep = torch.export._trace._export(
                model, example_inputs, dynamic_shapes=dynamic_shapes, pre_dispatch=True
            )
            gm = ep.module()
        else:
            with torch._export.config.patch(use_new_tracer_experimental=True):
                gm = torch.export._trace._export_to_torch_ir(
                    model,
                    example_inputs,
                    dynamic_shapes=dynamic_shapes,
                    disable_constraint_solver=disable_constraint_solver,
                    # Disabling this flag, because instead we can rely on the mapping
                    # dynamo_flat_name_to_original_fqn which is coming from Dynamo.
                    restore_fqn=False,
                )

        if IS_FBCODE:
            from deeplearning.aot_inductor.extern_node_thrift_serializer import (
                thrift_serializer,
            )

            if options is None:
                options = {}
            options["extern_node_serializer"] = thrift_serializer

        with torch.no_grad():
            so_path = torch._inductor.aot_compile(gm, example_inputs, options=options)  # type: ignore[arg-type]

        return so_path

    @staticmethod
    def legacy_load_runner(device, so_path: str) -> "AOTIModelContainerRunner":
        # [FUNCTION DOC] HELPER: Creates a model container runner for a compiled library. In FB
        # code the library (and a matching json of extern kernel nodes, if present) is copied to
        # a fresh temporary directory first to avoid stale dynamic-loading handles, and the
        # pybind runner is returned. Otherwise returns the CPU, XPU, MPS or CUDA runner
        # depending on the device string (any other value is treated as CUDA), each created with
        # one model instance.
        if IS_FBCODE:
            from .fb import test_aot_inductor_model_runner_pybind  # @manual

            with tempfile.TemporaryDirectory() as temp_dir:
                # copy *.so file to a unique path just before loading
                # to avoid stale dlopen handles when an updated *.so
                # from the same path is loaded repetitively in a test
                temp_so_path = os.path.join(temp_dir, "model.so")
                shutil.copy(so_path, temp_so_path)

                # We also need to copy over the serialized extern_kernel_nodes for custom ops
                extern_kernel_nodes_path = f"{so_path[:-3]}.json"
                if os.path.isfile(extern_kernel_nodes_path):
                    temp_extern_kernel_nodes_path = os.path.join(temp_dir, "model.json")
                    shutil.copy(extern_kernel_nodes_path, temp_extern_kernel_nodes_path)

                return test_aot_inductor_model_runner_pybind.Runner(
                    temp_so_path, device == "cpu"
                )
        else:
            if device == "cpu":
                return torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)
            elif device == "xpu":
                return torch._C._aoti.AOTIModelContainerRunnerXpu(so_path, 1, device)
            elif device == "mps":
                return torch._C._aoti.AOTIModelContainerRunnerMps(so_path, 1)
            else:
                return torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)

    @staticmethod
    def legacy_load(device, so_path):
        # [FUNCTION DOC] HELPER: Loads a compiled library as a callable. In FB code it builds a
        # runner and returns a function that reads the stored call spec, flattens args and kwargs
        # by the input tree spec, keeps only tensors, runs the runner and rebuilds the output by
        # the output tree spec. Elsewhere it returns the result of the export module loader for
        # the path and device. A TODO notes the two paths should be unified.
        # TODO: unify fbcode and oss behavior to only use torch._export.aot_load
        if IS_FBCODE:
            runner = AOTIRunnerUtil.legacy_load_runner(device, so_path)

            def optimized(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Callable returned for FB code: flattens inputs using the
            # saved call spec, filters to tensors, runs the runner and unflattens outputs into the
            # original structure.
                call_spec = runner.get_call_spec()
                in_spec = pytree.treespec_loads(call_spec[0])
                out_spec = pytree.treespec_loads(call_spec[1])
                flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
                flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
                flat_outputs = runner.run(flat_inputs)
                return pytree.tree_unflatten(flat_outputs, out_spec)

            return optimized
        else:
            return torch._export.aot_load(so_path, device)

    @staticmethod
    def legacy_run(
        device: str,
        model,
        example_inputs,
        options=None,
        dynamic_shapes=None,
        disable_constraint_solver=False,
    ):
        # [FUNCTION DOC] HELPER: Compiles the model through the legacy path, loads the library for
        # the device, and returns the result of calling it on the example inputs. Options, dynamic
        # shapes and the constraint solver flag are passed through to compile.
        so_path = AOTIRunnerUtil.legacy_compile(
            model,
            example_inputs,
            options=options,
            dynamic_shapes=dynamic_shapes,
            disable_constraint_solver=disable_constraint_solver,
        )
        optimized = AOTIRunnerUtil.legacy_load(device, so_path)
        return optimized(*example_inputs)

    @staticmethod
    def compile(
        model: torch.nn.Module | types.FunctionType,
        example_inputs: tuple[torch.Tensor, ...],
        inductor_configs: dict[str, Any] | None = None,
        dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None,
    ):
        # [FUNCTION DOC] HELPER: Exports the model (wrapping non-modules) in strict mode with
        # deferred runtime asserts preferred over guards, under no-grad with the new tracer config
        # patched, then packages it with AOTInductor using the given inductor configs. Returns the
        # path to the produced package.
        if not isinstance(model, torch.nn.Module):
            # This should really be the default behavior of torch.export.export
            model = WrapperModule(model)

        with (
            torch.no_grad(),
            torch._export.config.patch(use_new_tracer_experimental=True),
        ):
            # strict=False needs extra migration work
            ep = torch.export.export(
                model,
                example_inputs,
                dynamic_shapes=dynamic_shapes,
                strict=True,
                prefer_deferred_runtime_asserts_over_guards=True,
            )
            package_path = torch._inductor.aoti_compile_and_package(
                ep, inductor_configs=inductor_configs
            )
        return package_path

    @staticmethod
    def run(
        model: torch.nn.Module | types.FunctionType,
        example_inputs: tuple[torch.Tensor, ...],
        inductor_configs: dict[str, Any] | None = None,
        dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None,
    ):
        # [FUNCTION DOC] HELPER: Compiles the model to a package, loads the package and returns
        # the result of calling the loaded model on the example inputs.
        package_path = AOTIRunnerUtil.compile(
            model,
            example_inputs,
            inductor_configs=inductor_configs,
            dynamic_shapes=dynamic_shapes,
        )
        optimized = torch._inductor.aoti_load_package(package_path)
        return optimized(*example_inputs)

    @staticmethod
    def run_multiple(
        model: torch.nn.Module | types.FunctionType,
        list_example_inputs: list[tuple[torch.Tensor, ...]],
        inductor_configs: dict[str, Any] | None = None,
        dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None,
    ):
        # [FUNCTION DOC] HELPER: Compiles a package using the first set of example inputs, loads
        # it, then calls it once for every set of inputs and returns the list of outputs in order.
        package_path = AOTIRunnerUtil.compile(
            model,
            list_example_inputs[0],
            inductor_configs=inductor_configs,
            dynamic_shapes=dynamic_shapes,
        )
        optimized = torch._inductor.aoti_load_package(package_path)
        list_output_tensors = []
        for example_inputs in list_example_inputs:
            list_output_tensors.append(optimized(*example_inputs))
        return list_output_tensors


def check_model(
    self: TestCase,
    model,
    example_inputs,
    options=None,
    dynamic_shapes=None,
    atol=None,
    rtol=None,
    move_model_to_device=True,
):
    # [FUNCTION DOC] HELPER: Check method for test classes (first parameter is the test case).
    # Under no-grad and with the stack-allocation and minimal-arrayref options taken from the test
    # case, it seeds the random generator, optionally moves the model to the test device, and if
    # all tensor inputs are on cpu clones them to the test device preserving strides and offset.
    # It runs a deep copy of the model on copied inputs as the reference, reseeds, runs the
    # AOTInductor path, and asserts the outputs are equal with the given tolerances. The options
    # argument is forwarded as the inductor configs.
    with (
        torch.no_grad(),
        config.patch(
            {
                "aot_inductor.allow_stack_allocation": self.allow_stack_allocation,
                "aot_inductor.use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,
            }
        ),
    ):
        torch.manual_seed(0)
        if not isinstance(model, types.FunctionType) and move_model_to_device:
            model = model.to(self.device)

        # For non mixed device inputs with default "cpu",set the device manually.
        if all(
            t.device.type == "cpu"
            for t in example_inputs
            if isinstance(t, torch.Tensor)
        ):
            example_inputs = tuple(
                clone_preserve_strides_offset(x, device=self.device)
                for x in example_inputs
            )

        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(example_inputs)
        expected = ref_model(*ref_inputs)

        torch.manual_seed(0)
        actual = AOTIRunnerUtil.run(
            model,
            example_inputs,
            options,
            dynamic_shapes,
        )

    self.assertEqual(actual, expected, atol=atol, rtol=rtol)


def check_model_with_multiple_inputs(
    self: TestCase,
    model,
    list_example_inputs,
    options=None,
    dynamic_shapes=None,
):
    # [FUNCTION DOC] HELPER: Check method like the single-input one but for a list of input
    # tuples. Moves the model to the test device, computes eager reference outputs on copied
    # inputs, runs the compiled package on every input set, and asserts the output lists match
    # according to the dynamo same-comparison helper.
    with (
        torch.no_grad(),
        config.patch(
            {
                "aot_inductor.allow_stack_allocation": self.allow_stack_allocation,
                "aot_inductor.use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,
            }
        ),
    ):
        torch.manual_seed(0)
        model = model.to(self.device)
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(list_example_inputs)
        list_expected = [ref_model(*inputs) for inputs in ref_inputs]

        torch.manual_seed(0)
        list_actual = AOTIRunnerUtil.run_multiple(
            model, list_example_inputs, options, dynamic_shapes
        )

    self.assertTrue(same(list_actual, list_expected))


def code_check_count(
    self: TestCase,
    model,
    example_inputs,
    target_str: str,
    target_count: int,
):
    # [FUNCTION DOC] HELPER: Check method that AOT compiles the model with the stack-allocation
    # and minimal-arrayref options from the test case, reads the generated C++ file next to the
    # produced path, and asserts the target string appears exactly the target number of times.
    with (
        torch.no_grad(),
        config.patch(
            {
                "aot_inductor.allow_stack_allocation": self.allow_stack_allocation,
                "aot_inductor.use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,
            }
        ),
    ):
        package_path = torch._export.aot_compile(model, example_inputs)

    with open(os.path.splitext(package_path)[0] + ".cpp") as cpp:
        src_code = cpp.read()
        FileCheck().check_count(
            target_str,
            target_count,
            exactly=True,
        ).run(src_code)


if __name__ == "__main__":
    run_tests()
