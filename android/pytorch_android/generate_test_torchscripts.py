# [MODULE DOC] Script that builds the TorchScript test model used by the Android tests.
# Defines a scripted module with many small methods (identity functions, list and tuple
# reductions, memory-format and conv helpers) and saves it for the lite interpreter.
# Import-time side effects: scripts the module, prints its graph and saves the file
# test.pt under the relative output directory, so it must be run from the right folder.
import torch
from torch import Tensor


OUTPUT_DIR = "src/androidTest/assets/"


def scriptAndSave(module, fileName):
    # [FUNCTION DOC] HELPER: Script the given module, print a separator and the graph, save it
    # in lite-interpreter format to the output directory plus the given file name, and print
    # the saved path. Side effects: writes a file and prints to standard output.
    print("-" * 80)
    script_module = torch.jit.script(module)
    print(script_module.graph)
    outputFileName = OUTPUT_DIR + fileName
    # note that the lite interpreter model can also be used in full JIT
    script_module._save_for_lite_interpreter(outputFileName)
    print("Saved to " + outputFileName)
    print("=" * 80)


class Test(torch.jit.ScriptModule):
    # [CLASS DOC] Script module holding the methods exercised by the Android tests. Each
    # method is compiled with TorchScript; most echo or transform their inputs so the tests
    # can compare Java-side results. The module has no state.
    @torch.jit.script_method
    def forward(self, input):
        # [FUNCTION DOC] Default forward method; ignores its input and returns None.
        return None

    @torch.jit.script_method
    def eqBool(self, input: bool) -> bool:
        # [FUNCTION DOC] Return the boolean input unchanged.
        return input

    @torch.jit.script_method
    def eqInt(self, input: int) -> int:
        # [FUNCTION DOC] Return the integer input unchanged.
        return input

    @torch.jit.script_method
    def eqFloat(self, input: float) -> float:
        # [FUNCTION DOC] Return the float input unchanged.
        return input

    @torch.jit.script_method
    def eqStr(self, input: str) -> str:
        # [FUNCTION DOC] Return the string input unchanged.
        return input

    @torch.jit.script_method
    def eqTensor(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Return the tensor input unchanged.
        return input

    @torch.jit.script_method
    def eqDictStrKeyIntValue(self, input: dict[str, int]) -> dict[str, int]:
        # [FUNCTION DOC] Return the string-to-int dictionary input unchanged.
        return input

    @torch.jit.script_method
    def eqDictIntKeyIntValue(self, input: dict[int, int]) -> dict[int, int]:
        # [FUNCTION DOC] Return the int-to-int dictionary input unchanged.
        return input

    @torch.jit.script_method
    def eqDictFloatKeyIntValue(self, input: dict[float, int]) -> dict[float, int]:
        # [FUNCTION DOC] Return the float-to-int dictionary input unchanged.
        return input

    @torch.jit.script_method
    def listIntSumReturnTuple(self, input: list[int]) -> tuple[list[int], int]:
        # [FUNCTION DOC] Sum the integers of the input list and return a tuple of the original
        # list and the sum (zero for an empty list).
        sum = 0
        for x in input:
            sum += x
        return (input, sum)

    @torch.jit.script_method
    def listBoolConjunction(self, input: list[bool]) -> bool:
        # [FUNCTION DOC] Return the logical AND of all booleans in the list (true if empty).
        res = True
        for x in input:
            res = res and x
        return res

    @torch.jit.script_method
    def listBoolDisjunction(self, input: list[bool]) -> bool:
        # [FUNCTION DOC] Return the logical OR of all booleans in the list (false if empty).
        res = False
        for x in input:
            res = res or x
        return res

    @torch.jit.script_method
    def tupleIntSumReturnTuple(
        self, input: tuple[int, int, int]
    ) -> tuple[tuple[int, int, int], int]:
        # [FUNCTION DOC] Sum the three integers of the input tuple and return a tuple of the
        # original tuple and the sum.
        sum = 0
        for x in input:
            sum += x
        return (input, sum)

    @torch.jit.script_method
    def optionalIntIsNone(self, input: int | None) -> bool:
        # [FUNCTION DOC] Return whether the optional integer input is absent.
        return input is None

    @torch.jit.script_method
    def intEq0None(self, input: int) -> int | None:
        # [FUNCTION DOC] Return None when the integer input is zero, else return it unchanged.
        if input == 0:
            return None
        return input

    @torch.jit.script_method
    def str3Concat(self, input: str) -> str:
        # [FUNCTION DOC] Return the input string repeated three times by concatenation.
        return input + input + input

    @torch.jit.script_method
    def newEmptyShapeWithItem(self, input):
        # [FUNCTION DOC] Convert the single value of the input tensor to an integer, build a
        # one-element tensor from it and return its first element (a zero-dim tensor).
        return torch.tensor([int(input.item())])[0]

    @torch.jit.script_method
    def testAliasWithOffset(self) -> list[Tensor]:
        # [FUNCTION DOC] Build a two-element tensor and return a list holding its two element
        # views, the second sharing storage at a nonzero offset (tests aliasing with offset).
        x = torch.tensor([100, 200])
        a = [x[0], x[1]]
        return a

    @torch.jit.script_method
    def testNonContiguous(self):
        # [FUNCTION DOC] Take every second element of a three-element tensor, giving a
        # non-contiguous view. Raises an assertion error if the view is contiguous or its two
        # elements are not the first and third values. Returns the view.
        x = torch.tensor([100, 200, 300])[::2]
        if x.is_contiguous():
            raise AssertionError("expected non-contiguous tensor")
        if x[0] != 100:
            raise AssertionError("expected x[0] == 100")
        if x[1] != 300:
            raise AssertionError("expected x[1] == 300")
        return x

    @torch.jit.script_method
    def conv2d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        # [FUNCTION DOC] Run a 2D convolution of x with weights w, then make the result
        # contiguous in channels-last format if requested, otherwise in default format.
        r = torch.nn.functional.conv2d(x, w)
        if toChannelsLast:
            r = r.contiguous(memory_format=torch.channels_last)
        else:
            r = r.contiguous()
        return r

    @torch.jit.script_method
    def conv3d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        # [FUNCTION DOC] Run a 3D convolution of x with weights w, then make the result
        # contiguous in the 3D channels-last format if requested, otherwise in default format.
        r = torch.nn.functional.conv3d(x, w)
        if toChannelsLast:
            r = r.contiguous(memory_format=torch.channels_last_3d)
        else:
            r = r.contiguous()
        return r

    @torch.jit.script_method
    def contiguous(self, x: Tensor) -> Tensor:
        # [FUNCTION DOC] Return the input tensor made contiguous in the default format.
        return x.contiguous()

    @torch.jit.script_method
    def contiguousChannelsLast(self, x: Tensor) -> Tensor:
        # [FUNCTION DOC] Return the input tensor made contiguous in channels-last format.
        return x.contiguous(memory_format=torch.channels_last)

    @torch.jit.script_method
    def contiguousChannelsLast3d(self, x: Tensor) -> Tensor:
        # [FUNCTION DOC] Return the input tensor made contiguous in the 3D channels-last format.
        return x.contiguous(memory_format=torch.channels_last_3d)


scriptAndSave(Test(), "test.pt")
