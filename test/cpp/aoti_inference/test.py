# [MODULE DOC] Script that generates AOTI test data for the C++ inference tests.
# Defines two small models (a matmul net with weights, and one with cuda tensor constants),
# AOT-compiles them (shared library and .pt2 package) for cpu/cuda, with and without runtime
# constant folding, and runs the reference models. Results are collected into module-level
# mappings and saved as scripted containers: data.pt, large_data.pt, data_with_tensor_constants.pt
# and cuda_alloc_data.pt in the current directory. Import-time side effects: seeds the RNG, runs
# all generators immediately (no main guard), compiles models and writes files.
import torch
import torch._inductor.config
from torch._export import aot_compile
from torch.export import Dim


torch.manual_seed(1337)


class Net(torch.nn.Module):
    # [CLASS DOC] HELPER: Model with two random square weight matrices w_pre and w_add created on
    # the given device; forward is a matmul of the input with relu(w_pre transposed) plus w_add.
    def __init__(self, device, size=4):
        # [FUNCTION DOC] HELPER: Creates random weights w_pre, w_add (size by size) on the device.
        super().__init__()
        self.w_pre = torch.randn(size, size, device=device)
        self.w_add = torch.randn(size, size, device=device)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns x matrix-multiplied with relu(transpose of w_pre) plus
        # w_add.
        w_transpose = torch.transpose(self.w_pre, 0, 1)
        w_relu = torch.nn.functional.relu(w_transpose)
        w = w_relu + self.w_add
        return torch.matmul(x, w)


class NetWithTensorConstants(torch.nn.Module):
    # [CLASS DOC] HELPER: Model holding a random cuda tensor constant w (30 by 1); forward
    # multiplies it with both inputs and returns a fixed selection of rows (indices 0..15 and 17).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the cuda tensor constant w; requires a cuda device.
        super().__init__()
        self.w = torch.randn(30, 1, device="cuda")

    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Returns the product of w, x and y, indexed to the selected 17
        # rows.
        z = self.w * x * y
        return z[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]]


data = {}
large_data = {}
cuda_alloc_data = {}
data_with_tensor_constants = {}


# Basice AOTI model test generation.
def generate_basic_tests():
    # [FUNCTION DOC] HELPER: Generates basic test data for cpu (and cuda if available); runtime
    # constant folding is tried only on cuda. For each combination, compiles the model to a shared
    # library and a .pt2 package with a dynamic batch dimension, and stores paths, inputs,
    # reference outputs and weights into the global data mapping under suffixed keys. Mutates
    # global state.
    for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
        for use_runtime_constant_folding in [True, False]:
            if device == "cpu" and use_runtime_constant_folding:
                # We do not test runtime const folding for cpu mode.
                continue
            model = Net(device).to(device=device)
            x = torch.randn((4, 4), device=device)
            with torch.no_grad():
                ref_output = model(x)

            torch._dynamo.reset()
            with torch.no_grad():
                dim0_x = Dim("dim0_x", min=1, max=1024)
                dynamic_shapes = {"x": {0: dim0_x}}
                model_so_path = aot_compile(
                    model,
                    (x,),
                    dynamic_shapes=dynamic_shapes,
                    options={
                        "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                    },
                )
                # Also store a .pt2 file using the aoti_compile_and_package API
                pt2_package_path = torch._inductor.aoti_compile_and_package(
                    torch.export.export(
                        model,
                        (x,),
                        dynamic_shapes=dynamic_shapes,
                    ),
                    inductor_configs={
                        "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                    },
                )

            suffix = f"{device}"
            if use_runtime_constant_folding:
                suffix += "_use_runtime_constant_folding"
            data.update(
                {
                    f"model_so_path_{suffix}": model_so_path,
                    f"pt2_package_path_{suffix}": pt2_package_path,
                    f"inputs_{suffix}": [x],
                    f"outputs_{suffix}": [ref_output],
                    f"w_pre_{suffix}": model.w_pre,
                    f"w_add_{suffix}": model.w_add,
                }
            )


def generate_basic_tests_consts_cpp():
    # [FUNCTION DOC] HELPER: Reruns the basic generation with consts-asm build disabled in the
    # inductor config (to test the consts cpp path), then restores the previous setting.
    # Updates the global data mapping.
    backup_consts_asm_cfg: bool = (
        torch._inductor.config.aot_inductor.use_consts_asm_build
    )
    torch._inductor.config.aot_inductor.use_consts_asm_build = False

    # Test consts cpp build again.
    generate_basic_tests()

    torch._inductor.config.aot_inductor.use_consts_asm_build = backup_consts_asm_cfg


def generate_large_tests():
    # [FUNCTION DOC] HELPER: Generates large (4096 square) cuda test data, compiled with and
    # without runtime constant folding; stores paths, input, output and weights in large_data.
    device = "cuda"
    model = Net(device, size=4096).to(device=device)
    x = torch.randn((4096, 4096), device=device)
    with torch.no_grad():
        ref_output = model(x)

    torch._dynamo.reset()
    for use_runtime_constant_folding in [True, False]:
        with torch.no_grad():
            model_so_path = aot_compile(
                model,
                (x,),
                options={
                    "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                },
            )
            # Also store a .pt2 file using the aoti_compile_and_package API
            pt2_package_path = torch._inductor.aoti_compile_and_package(
                torch.export.export(
                    model,
                    (x,),
                ),
                inductor_configs={
                    "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                },
            )

        suffix = "_use_runtime_constant_folding" if use_runtime_constant_folding else ""
        large_data.update(
            {
                f"model_so_path{suffix}": model_so_path,
                f"pt2_package_path{suffix}": pt2_package_path,
                "inputs": [x],
                "outputs": [ref_output],
                "w_pre": model.w_pre,
                "w_add": model.w_add,
            }
        )


def generate_cuda_alloc_test():
    # [FUNCTION DOC] HELPER: Generates cuda test data compiled with the weight caching allocator
    # option; stores path, input, output and weights in cuda_alloc_data. Requires cuda.
    device = "cuda"
    model = Net(device, size=4096).to(device=device)
    x = torch.randn((4096, 4096), device=device)
    with torch.no_grad():
        ref_output = model(x)

    torch._dynamo.reset()
    with torch.no_grad():
        model_so_path = aot_compile(
            model,
            (x,),
            options={"aot_inductor.weight_use_caching_allocator": True},
        )

    cuda_alloc_data.update(
        {
            "model_so_path": model_so_path,
            "inputs": [x],
            "outputs": [ref_output],
            "w_pre": model.w_pre,
            "w_add": model.w_add,
        }
    )


# AOTI model which will create additional tensors during autograd.
def generate_test_with_additional_tensors():
    # [FUNCTION DOC] HELPER: Does nothing without cuda. Otherwise compiles the tensor-constant
    # model to a library and a .pt2 package and stores results in data_with_tensor_constants.
    if not torch.cuda.is_available():
        return

    model = NetWithTensorConstants()
    x = torch.randn((30, 1), device="cuda")
    y = torch.randn((30, 1), device="cuda")
    with torch.no_grad():
        ref_output = model(x, y)

    torch._dynamo.reset()
    with torch.no_grad():
        model_so_path = aot_compile(model, (x, y))
        # Also store a .pt2 file using the aoti_compile_and_package API
        pt2_package_path = torch._inductor.aoti_compile_and_package(
            torch.export.export(model, (x, y))
        )

    data_with_tensor_constants.update(
        {
            "model_so_path": model_so_path,
            "pt2_package_path": pt2_package_path,
            "inputs": [x, y],
            "outputs": [ref_output],
            "w": model.w,
        }
    )


generate_basic_tests()
generate_basic_tests_consts_cpp()
generate_large_tests()
generate_test_with_additional_tensors()
generate_cuda_alloc_test()


# Use this to communicate tensors to the cpp code
class Serializer(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose attributes are set from a mapping so tensors can be scripted
    # and saved for the C++ tests.
    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Sets each key of data as a module attribute holding its value.
        super().__init__()
        for key in data:
            setattr(self, key, data[key])


torch.jit.script(Serializer(data)).save("data.pt")
torch.jit.script(Serializer(large_data)).save("large_data.pt")
torch.jit.script(Serializer(data_with_tensor_constants)).save(
    "data_with_tensor_constants.pt"
)
torch.jit.script(Serializer(cuda_alloc_data)).save("cuda_alloc_data.pt")
