# [MODULE DOC] HELPER: Code generators that produce the text of a C++ main.cpp driver and a CMakeLists file for
# standalone AOTInductor packages. Exports nothing publicly (empty __all__). Import-time side effect: creates a logger.
import logging

import torch
from torch._inductor.utils import IndentedBuffer


__all__ = []  # type: ignore[var-annotated]
logger = logging.getLogger(__name__)


def _get_main_cpp_file(
    package_name: str,
    model_names: list[str],
    example_inputs_map: dict[str, int] | None,
    device_type: str,
) -> str:
    # [FUNCTION DOC] HELPER: Returns C++ source text for main.cpp. It includes headers (CUDA or HIP ones by device_type),
    # per-model AOTInductor headers, and a main that creates the device, constructs and loads each model's constants,
    # and, only if example_inputs_map (model name to input count) is given, loads pickled input tensors from files,
    # runs each model, prints and saves outputs. Errors are caught in the generated code and return 1. Pure text building.
    # KNOWN ISSUE: the docstring documents a cuda argument that does not exist and mis-types example_inputs_map.
    """
    Generates a main.cpp file for AOTInductor standalone models in the specified package.

    Args:
        package_name (str): Name of the package containing the models.
        model_names (List[str]): List of model names to include in the generated main.cpp.
        cuda (bool): Whether to generate code with CUDA support.
        example_inputs_map (Optional[Dict[str, List[Tensor]]]): A mapping from model name to
            its list of example input tensors. If provided, the generated main.cpp will
            load and run these inputs.

    Returns:
        str: The contents of the generated main.cpp file as a string.
    """

    ib = IndentedBuffer()

    ib.writelines(
        [
            "#include <dlfcn.h>",
            "#include <fstream>",
            "#include <iostream>",
            "#include <memory>",
            "#include <torch/torch.h>",
            "#include <vector>",
            "#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>",
        ]
    )
    if device_type == "cuda":
        if torch.version.hip:
            ib.writelines(
                [
                    "#include <hip/hip_runtime.h>",
                ]
            )

        else:
            ib.writelines(
                [
                    "#include <cuda.h>",
                    "#include <cuda_runtime_api.h>",
                ]
            )
    for model_name in model_names:
        ib.writeline(
            f'#include "{package_name}/data/aotinductor/{model_name}/{model_name}.h"'
        )

    ib.newline()
    for model_name in model_names:
        ib.writeline(f"using torch::aot_inductor::AOTInductorModel{model_name};")

    ib.writelines(
        [
            "using torch::aot_inductor::ConstantHandle;",
            "using torch::aot_inductor::ConstantMap;",
            "",
            "int main(int argc, char* argv[]) {",
        ]
    )

    with ib.indent():
        ib.writeline(f'std::string device_str = "{device_type}";')
        ib.writeline("try {")

        with ib.indent():
            ib.writeline("c10::Device device(device_str);")

            if example_inputs_map is not None:
                # TODO: add device
                for i, model_name in enumerate(model_names):
                    num_inputs = example_inputs_map[model_name]

                    ib.writeline(f"// Load input tensors for model {model_name}")
                    ib.writeline(f"std::vector<at::Tensor> input_tensors{i + 1};")
                    ib.writeline(f"for (int j = 0; j < {num_inputs}; ++j) {{")
                    with ib.indent():
                        ib.writeline(
                            f'std::string filename = "{model_name}_input_" + std::to_string(j) + ".pt";'
                        )
                        ib.writeline("std::ifstream in(filename, std::ios::binary);")
                        ib.writeline("if (!in.is_open()) {")
                        with ib.indent():
                            ib.writeline(
                                'std::cerr << "Failed to open file: " << filename << std::endl;'
                            )
                            ib.writeline("return 1;")
                        ib.writeline("}")
                        ib.writeline(
                            "std::vector<char> buffer((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());"
                        )
                        ib.writeline(
                            "torch::IValue ivalue = torch::pickle_load(buffer);"
                        )
                        ib.writeline(
                            f"input_tensors{i + 1}.push_back(ivalue.toTensor().to(device));"
                        )
                    ib.writeline("}")
                    ib.newline()

                ib.newline()
                ib.writeline("\n// Create array of input handles")
                for i in range(len(model_names)):
                    ib.writelines(
                        [
                            f"auto input_handles{i + 1} =",
                            f"    torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(input_tensors{i + 1});",
                        ]
                    )

                ib.writeline("\n// Create array for output handles")
                for i in range(len(model_names)):
                    ib.writeline(f"AtenTensorHandle output_handle{i + 1};")

            ib.writeline("\n// Create and load models")
            for i, model_name in enumerate(model_names):
                ib.writelines(
                    [
                        f"auto constants_map{i + 1} = std::make_shared<ConstantMap>();",
                        f"auto constants_array{i + 1} = std::make_shared<std::vector<ConstantHandle>>();",
                        f"auto model{i + 1} = std::make_unique<AOTInductorModel{model_name}>(",
                        f"    std::move(constants_map{i + 1}),",
                        f"    std::move(constants_array{i + 1}),",
                        "    device_str,",
                        f'    "{package_name}/data/aotinductor/{model_name}/");',
                        f"model{i + 1}->load_constants();",
                    ]
                )

            if example_inputs_map is not None:
                ib.writeline("\n// Run the models")
                for i in range(len(model_names)):
                    ib.writeline(
                        f"torch::aot_inductor::DeviceStreamType stream{i + 1} = nullptr;"
                    )
                    ib.writeline(
                        f"model{i + 1}->run(&input_handles{i + 1}[0], &output_handle{i + 1}, stream{i + 1}, nullptr);"
                    )

                ib.writeline("\n// Convert output handles to tensors")
                for i in range(len(model_names)):
                    ib.writelines(
                        [
                            f"auto output_tensor{i + 1} =",
                            f"    torch::aot_inductor::alloc_tensors_by_stealing_from_handles(&output_handle{i + 1}, 1);",
                        ]
                    )

                ib.writeline("\n// Validate outputs")
                for i in range(len(model_names)):
                    ib.writeline(
                        f"""std::cout << "output_tensor{i + 1}\\n" << output_tensor{i + 1} << std::endl;"""
                    )
                    ib.writeline(
                        f"""torch::save(output_tensor{i + 1}, "output_tensor{i + 1}.pt");"""
                    )

            ib.writeline("return 0;")

        ib.writelines(
            [
                "} catch (const std::exception &e) {",
            ]
        )
        with ib.indent():
            ib.writeline('std::cerr << "Error: " << e.what() << std::endl;')
            ib.writeline("return 1;")

        ib.writeline("}")
    ib.writeline("}")

    return ib.getvalue()


def _get_make_file(package_name: str, model_names: list[str], device_type: str) -> str:
    # [FUNCTION DOC] HELPER: Returns CMakeLists text that builds an executable from main.cpp, adds each model's
    # subdirectory, finds Torch (when test_configs.use_libtorch is set) and CUDA or HIP packages, defines USE_CUDA,
    # USE_HIP or USE_XPU per device_type, and links torch, the model libraries and device libraries. Pure text building.
    ib = IndentedBuffer()

    ib.writelines(
        [
            "cmake_minimum_required(VERSION 3.10)",
            "project(TestProject)",
            "",
            "set(CMAKE_CXX_STANDARD 20)",
            "",
        ]
    )

    from torch._inductor.config import test_configs

    if test_configs.use_libtorch:
        ib.writeline("find_package(Torch REQUIRED)")

    if device_type == "cuda":
        if torch.version.hip:
            ib.writeline("find_package(hip REQUIRED)")
        else:
            ib.writeline("find_package(CUDA REQUIRED)")

    ib.newline()
    for model_name in model_names:
        ib.writeline(f"add_subdirectory({package_name}/data/aotinductor/{model_name}/)")

    ib.writeline("\nadd_executable(main main.cpp)")
    if device_type == "cuda":
        if torch.version.hip:
            ib.writeline("target_compile_definitions(main PRIVATE USE_HIP)")
        else:
            ib.writeline("target_compile_definitions(main PRIVATE USE_CUDA)")
    elif device_type == "xpu":
        ib.writeline("target_compile_definitions(main PRIVATE USE_XPU)")

    model_libs = " ".join(model_names)
    ib.writeline(f"target_link_libraries(main PRIVATE torch {model_libs})")

    if device_type == "cuda":
        if torch.version.hip:
            ib.writeline("target_link_libraries(main PRIVATE hip::host)")
        else:
            ib.writeline("target_link_libraries(main PRIVATE cuda ${CUDA_LIBRARIES})")
    elif device_type == "xpu":
        ib.writeline("target_link_libraries(main PRIVATE sycl ze_loader)")
    return ib.getvalue()
