# [MODULE DOC] Defines a TorchScript module of small methods used by the Android API tests (mobile
# model test assets). Each method has a simple, deterministic behavior (identity on a typed value,
# simple list or tuple reduction, memory format conversion) so that the Java side can check
# argument and result conversion. No import-time side effects beyond class creation.
import torch
from torch import Tensor


class AndroidAPIModule(torch.jit.ScriptModule):
    # [CLASS DOC] Script module with many script methods that mostly echo or reduce their typed
    # arguments. It has no state. Methods cover bool, int, float, string, tensor, dictionaries with
    # string, int and float keys, lists, tuples, optionals, tensor views and contiguity.
    @torch.jit.script_method
    def forward(self, input):
        # [FUNCTION DOC] Ignores its input and returns nothing.
        return None

    @torch.jit.script_method
    def eqBool(self, input: bool) -> bool:
        # [FUNCTION DOC] Returns the boolean argument unchanged.
        return input

    @torch.jit.script_method
    def eqInt(self, input: int) -> int:
        # [FUNCTION DOC] Returns the integer argument unchanged.
        return input

    @torch.jit.script_method
    def eqFloat(self, input: float) -> float:
        # [FUNCTION DOC] Returns the float argument unchanged.
        return input

    @torch.jit.script_method
    def eqStr(self, input: str) -> str:
        # [FUNCTION DOC] Returns the string argument unchanged.
        return input

    @torch.jit.script_method
    def eqTensor(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Returns the tensor argument unchanged.
        return input

    @torch.jit.script_method
    def eqDictStrKeyIntValue(self, input: dict[str, int]) -> dict[str, int]:
        # [FUNCTION DOC] Returns the dictionary with string keys and int values unchanged.
        return input

    @torch.jit.script_method
    def eqDictIntKeyIntValue(self, input: dict[int, int]) -> dict[int, int]:
        # [FUNCTION DOC] Returns the dictionary with int keys and int values unchanged.
        return input

    @torch.jit.script_method
    def eqDictFloatKeyIntValue(self, input: dict[float, int]) -> dict[float, int]:
        # [FUNCTION DOC] Returns the dictionary with float keys and int values unchanged.
        return input

    @torch.jit.script_method
    def listIntSumReturnTuple(self, input: list[int]) -> tuple[list[int], int]:
        # [FUNCTION DOC] Sums the integers in the list and returns a tuple of the original list and
        # the sum.
        sum = 0
        for x in input:
            sum += x
        return (input, sum)

    @torch.jit.script_method
    def listBoolConjunction(self, input: list[bool]) -> bool:
        # [FUNCTION DOC] Returns true if every boolean in the list is true (true for an
        # empty list).
        res = True
        for x in input:
            res = res and x
        return res

    @torch.jit.script_method
    def listBoolDisjunction(self, input: list[bool]) -> bool:
        # [FUNCTION DOC] Returns true if any boolean in the list is true (false for an empty list).
        res = False
        for x in input:
            res = res or x
        return res

    @torch.jit.script_method
    def tupleIntSumReturnTuple(
        self, input: tuple[int, int, int]
    ) -> tuple[tuple[int, int, int], int]:
        # [FUNCTION DOC] Sums the three integers in the tuple and returns a tuple of the original
        # tuple and the sum.
        sum = 0
        for x in input:
            sum += x
        return (input, sum)

    @torch.jit.script_method
    def optionalIntIsNone(self, input: int | None) -> bool:
        # [FUNCTION DOC] Returns true if the optional integer argument is absent.
        return input is None

    @torch.jit.script_method
    def intEq0None(self, input: int) -> int | None:
        # [FUNCTION DOC] Returns nothing when the argument is 0, otherwise returns the argument.
        if input == 0:
            return None
        return input

    @torch.jit.script_method
    def str3Concat(self, input: str) -> str:
        # [FUNCTION DOC] Returns the string repeated three times by concatenation.
        return input + input + input

    @torch.jit.script_method
    def newEmptyShapeWithItem(self, input):
        # [FUNCTION DOC] Converts the single item of the input tensor to an int, builds a
        # one-element tensor from it and returns its first element, a zero-dimensional tensor.
        return torch.tensor([int(input.item())])[0]

    @torch.jit.script_method
    def testAliasWithOffset(self) -> list[Tensor]:
        # [FUNCTION DOC] Builds a two-element tensor and returns a list of its two elements as
        # separate tensors (views into the same storage at different offsets).
        x = torch.tensor([100, 200])
        a = [x[0], x[1]]
        return a

    @torch.jit.script_method
    def testNonContiguous(self):
        # [FUNCTION DOC] Takes every second element of a three-element tensor, raises an assertion
        # error unless the result is non-contiguous with first value 100 and second value 300, and
        # returns it.
        x = torch.tensor([100, 200, 300])[::2]
        if x.is_contiguous():
            raise AssertionError("Expected tensor to be non-contiguous")
        if x[0] != 100:
            raise AssertionError(f"Expected x[0] == 100, got {x[0]}")
        if x[1] != 300:
            raise AssertionError(f"Expected x[1] == 300, got {x[1]}")
        return x

    @torch.jit.script_method
    def conv2d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        # [FUNCTION DOC] Applies a 2D convolution of x with weights w (no bias, default stride and
        # padding), then returns the result in channels-last memory format if the flag is set, else
        # in the default contiguous format.
        r = torch.nn.functional.conv2d(x, w)
        if toChannelsLast:
            r = r.contiguous(memory_format=torch.channels_last)
        else:
            r = r.contiguous()
        return r

    @torch.jit.script_method
    def conv3d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        # [FUNCTION DOC] Applies a 3D convolution of x with weights w (no bias, default settings),
        # then returns the result in channels-last 3D memory format if the flag is set, else in the
        # default contiguous format.
        r = torch.nn.functional.conv3d(x, w)
        if toChannelsLast:
            r = r.contiguous(memory_format=torch.channels_last_3d)
        else:
            r = r.contiguous()
        return r

    @torch.jit.script_method
    def contiguous(self, x: Tensor) -> Tensor:
        # [FUNCTION DOC] Returns the tensor made contiguous in the default memory format.
        return x.contiguous()

    @torch.jit.script_method
    def contiguousChannelsLast(self, x: Tensor) -> Tensor:
        # [FUNCTION DOC] Returns the tensor made contiguous in channels-last memory format.
        return x.contiguous(memory_format=torch.channels_last)

    @torch.jit.script_method
    def contiguousChannelsLast3d(self, x: Tensor) -> Tensor:
        # [FUNCTION DOC] Returns the tensor made contiguous in channels-last 3D memory format.
        return x.contiguous(memory_format=torch.channels_last_3d)
