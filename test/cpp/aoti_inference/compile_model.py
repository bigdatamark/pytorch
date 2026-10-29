# [MODULE DOC] Script that AOT-compiles a small linear+ReLU model for the AOTI inference C++ tests.
# For each available device (cpu, plus cuda if present) it compiles the model with a dynamic batch
# dimension, wraps the compiled library in a traced module using the custom class MyAOTIClass, and
# saves it as script_model_<device>.pt. Sample inputs and reference outputs are saved to
# script_data.pt. Import-time side effect: loads the shared library libaoti_custom_class.so
# (path relative to the working directory) into torch.classes; fails if it is missing.
import torch
from torch.export import Dim


# custom op that loads the aot-compiled model
AOTI_CUSTOM_OP_LIB = "libaoti_custom_class.so"
torch.classes.load_library(AOTI_CUSTOM_OP_LIB)


class TensorSerializer(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose attributes are set from a mapping, so that tensors/lists
    # can be scripted and saved. Each key of the data mapping becomes an attribute with the given
    # value.
    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Sets each key of data as a module attribute holding its value.
        super().__init__()
        for key in data:
            setattr(self, key, data[key])


class SimpleModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Tiny model (linear 4->6 then ReLU) that is compiled by the script.
    """
    a simple module to be compiled
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the linear layer (4 in, 6 out) and the ReLU activation.
        super().__init__()
        self.fc = torch.nn.Linear(4, 6)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the linear layer then ReLU to x and returns the result.
        a = self.fc(x)
        b = self.relu(a)
        return b


class MyAOTIModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Wrapper module holding one MyAOTIClass custom-class instance built from
    # the compiled library path and device; its forward delegates to that instance.
    """
    a wrapper nn.Module that instantiates its forward method
    on MyAOTIClass
    """

    def __init__(self, lib_path, device):
        # [FUNCTION DOC] HELPER: Instantiates the MyAOTIClass custom class from lib_path and
        # device and stores it as aoti_custom_op.
        super().__init__()
        self.aoti_custom_op = torch.classes.aoti.MyAOTIClass(
            lib_path,
            device,
        )

    def forward(self, *x):
        # [FUNCTION DOC] HELPER: Runs the custom op on the tuple of positional inputs and returns
        # the outputs as a tuple.
        outputs = self.aoti_custom_op.forward(x)
        return tuple(outputs)


def make_script_module(lib_path, device, *inputs):
    # [FUNCTION DOC] HELPER: Builds a MyAOTIModule for the compiled library, runs it once on the
    # inputs as a sanity check (errors propagate), and returns a traced script module of it.
    m = MyAOTIModule(lib_path, device)
    # sanity check
    m(*inputs)
    return torch.jit.trace(m, inputs)


def compile_model(device, data):
    # [FUNCTION DOC] HELPER: Compiles the simple model on the given device with a dynamic batch
    # dimension (1 to 1024), saves the scripted wrapper to a file named after the device, and
    # records the sample inputs and reference output in the data mapping under device-suffixed
    # keys (mutates data in place). Writes a file to the current directory; returns nothing.
    module = SimpleModule().to(device)
    x = torch.randn((4, 4), device=device)
    inputs = (x,)
    # make batch dimension
    batch_dim = Dim("batch", min=1, max=1024)
    dynamic_shapes = {
        "x": {0: batch_dim},
    }
    with torch.no_grad():
        # aot-compile the module into a .so pointed by lib_path
        lib_path = torch._export.aot_compile(
            module, inputs, dynamic_shapes=dynamic_shapes
        )
    script_module = make_script_module(lib_path, device, *inputs)
    aoti_script_model = f"script_model_{device}.pt"
    script_module.save(aoti_script_model)

    # save sample inputs and ref output
    with torch.no_grad():
        ref_output = module(*inputs)
    data.update(
        {
            f"inputs_{device}": list(inputs),
            f"outputs_{device}": [ref_output],
        }
    )


def main():
    # [FUNCTION DOC] HELPER: Entry point. Compiles for every available device, then scripts and
    # saves the collected data mapping to script_data.pt in the current directory.
    data = {}
    for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
        compile_model(device, data)
    torch.jit.script(TensorSerializer(data)).save("script_data.pt")


if __name__ == "__main__":
    main()
