# [MODULE DOC] Setup and teardown script for lightweight-dispatch mobile tests. Import-time effect:
# every decorated model definition is replaced by a function that, when called, scripts the model,
# serializes it for the lite interpreter, records the operators it uses in a module-level set and
# saves a model file named for the class in the current directory. When run as a program it takes a
# command (setup or shutdown) and an operator yaml path: setup calls each model function, backs up
# the yaml and appends the collected operators as list entries; shutdown tries to remove the saved
# model files and restores the yaml from the backup.
# KNOWN ISSUE: the shutdown branch iterates the list of model classes (empty in a fresh process,
#     and classes rather than paths), so it never deletes the saved model files.
import functools
import os
import shutil
import sys
from io import BytesIO

import torch
from torch.jit.mobile import _export_operator_list, _load_for_lite_interpreter


_OPERATORS = set()
_FILENAMES = []
_MODELS = []


def save_model(cls):
    # [FUNCTION DOC] HELPER: Class decorator. Returns a replacement function (keeping the class
    # metadata) that, when called with no arguments, records the class, instantiates it, scripts
    # it, round-trips it through a lite interpreter buffer to list its operators, adds them to the
    # module-level operator set, records and writes the model file named for the class with the ptl
    # suffix in the current directory. The decorated name therefore refers to this function.
    """Save a model and dump all the ops"""

    @functools.wraps(cls)
    def wrapper_save():
        # [FUNCTION DOC] HELPER: Replacement for the decorated class; performs the save and
        # operator export steps described for the decorator and returns nothing.
        _MODELS.append(cls)
        model = cls()
        scripted = torch.jit.script(model)
        buffer = BytesIO(scripted._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)
        ops = _export_operator_list(mobile_module)
        _OPERATORS.update(ops)
        path = f"./{cls.__name__}.ptl"
        _FILENAMES.append(path)
        scripted._save_for_lite_interpreter(path)

    return wrapper_save


@save_model
class ModelWithDTypeDeviceLayoutPinMemory(torch.nn.Module):
    # [CLASS DOC] HELPER: Model whose forward creates an integer ones tensor of shape 3 by the
    # argument with explicit dtype, layout, device and pin-memory options.
    def forward(self, x: int):
        # [FUNCTION DOC] HELPER: Returns the ones tensor of shape 3 by x with int64 dtype, strided
        # layout, CPU, unpinned.
        a = torch.ones(
            size=[3, x],
            dtype=torch.int64,
            layout=torch.strided,
            device="cpu",
            pin_memory=False,
        )
        return a


@save_model
class ModelWithTensorOptional(torch.nn.Module):
    # [CLASS DOC] HELPER: Model that builds a 2 by 2 tensor of zeros, fills three entries, and
    # indexes it with the argument, covering operators with optional tensor arguments.
    def forward(self, index):
        # [FUNCTION DOC] HELPER: Fills a zero matrix by element assignment and returns the entry or
        # row selected by index.
        a = torch.zeros(2, 2)
        a[0][1] = 1
        a[1][0] = 2
        a[1][1] = 3
        return a[index]


# gradient.scalarrayint(Tensor self, *, Scalar[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
@save_model
class ModelWithScalarList(torch.nn.Module):
    # [CLASS DOC] HELPER: Model exercising operators with a list of scalars argument (gradient
    # with spacing).
    def forward(self, a: int):
        # [FUNCTION DOC] HELPER: Builds a four element tensor and returns its gradient with spacing
        # given as a tensor when the argument is 0, or as a list holding one number when it is 1;
        # returns nothing for other values.
        values = torch.tensor(
            [4.0, 1.0, 1.0, 16.0],
        )
        if a == 0:
            return torch.gradient(
                values, spacing=torch.scalar_tensor(2.0, dtype=torch.float64)
            )
        elif a == 1:
            return torch.gradient(values, spacing=[torch.tensor(1.0).item()])


# upsample_linear1d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
@save_model
class ModelWithFloatList(torch.nn.Upsample):
    # [CLASS DOC] HELPER: Linear upsampling module with scale factor list of one float, covering
    # operators with a float list argument.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the parent constructor with scale factor 2.0 as a
        # one-element tuple, linear mode, no align corners and recomputed scale factor.
        super().__init__(
            scale_factor=(2.0,),
            mode="linear",
            align_corners=False,
            recompute_scale_factor=True,
        )


# index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
@save_model
class ModelWithListOfOptionalTensors(torch.nn.Module):
    # [CLASS DOC] HELPER: Model indexing a tensor with a list holding a tensor and an index,
    # covering index operators with a list of optional tensors.
    def forward(self, index):
        # [FUNCTION DOC] HELPER: Returns an element selection of a one-row tensor using a zero
        # tensor and the given index.
        values = torch.tensor([[4.0, 1.0, 1.0, 16.0]])
        return values[torch.tensor(0), index]


# conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1,
# int groups=1) -> Tensor
@save_model
class ModelWithArrayOfInt(torch.nn.Conv2d):
    # [CLASS DOC] HELPER: 2D convolution with 1 input channel, 2 output channels, kernel 2 by 2,
    # stride 1 and padding 1, covering operators with integer array arguments.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the convolution constructor with the fixed channel, kernel,
        # stride and padding values.
        super().__init__(1, 2, (2, 2), stride=(1, 1), padding=(1, 1))


# add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
# ones_like(Tensor self, *, ScalarType?, dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None,
# MemoryFormat? memory_format=None) -> Tensor
@save_model
class ModelWithTensors(torch.nn.Module):
    # [CLASS DOC] HELPER: Model adding a tensor to a ones tensor of the same shape.
    def forward(self, a):
        # [FUNCTION DOC] HELPER: Returns the input plus a tensor of ones shaped like it.
        b = torch.ones_like(a)
        return a + b


@save_model
class ModelWithStringOptional(torch.nn.Module):
    # [CLASS DOC] HELPER: Model using division with an optional string rounding mode and an
    # out tensor.
    def forward(self, b):
        # [FUNCTION DOC] HELPER: Divides the input by a scalar int tensor 3 into a
        # preallocated output tensor, then returns a list of the truncating division result
        # and that output tensor.
        a = torch.tensor(3, dtype=torch.int64)
        out = torch.empty(size=[1], dtype=torch.float)
        torch.div(b, a, out=out)
        return [torch.div(b, a, rounding_mode="trunc"), out]


@save_model
class ModelWithMultipleOps(torch.nn.Module):
    # [CLASS DOC] HELPER: Model with a ReLU and flatten sequential container, covering
    # several operators.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds a sequential container of ReLU and flatten layers.
        super().__init__()
        self.ops = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Sets the second row of the input to -2 in place, then returns the
        # sequential container applied to it.
        x[1] = -2
        return self.ops(x)


if __name__ == "__main__":
    command = sys.argv[1]
    ops_yaml = sys.argv[2]
    backup = ops_yaml + ".bak"
    if command == "setup":
        tests = [
            ModelWithDTypeDeviceLayoutPinMemory(),
            ModelWithTensorOptional(),
            ModelWithScalarList(),
            ModelWithFloatList(),
            ModelWithListOfOptionalTensors(),
            ModelWithArrayOfInt(),
            ModelWithTensors(),
            ModelWithStringOptional(),
            ModelWithMultipleOps(),
        ]
        shutil.copyfile(ops_yaml, backup)
        with open(ops_yaml, "a") as f:
            f.writelines(f"- {op}\n" for op in _OPERATORS)
    elif command == "shutdown":
        for file in _MODELS:
            if os.path.isfile(file):
                os.remove(file)
        shutil.move(backup, ops_yaml)
