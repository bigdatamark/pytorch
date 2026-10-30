# Owner(s): ["oncall: quantization"]
# [MODULE DOC] Tests for quantization utilities: get_fqn_to_example_inputs, which records
# example inputs per submodule path, and _quantize_weight clamping to observer quant_min and
# quant_max for per-tensor and per-channel observers, plus a dtype smoke test for uint4 and
# int4 wrapper tensor subclasses. Running the file directly calls raise_on_run_directly.

import torch
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase
from torch.ao.quantization.utils import get_fqn_to_example_inputs
from torch.ao.nn.quantized.modules.utils import _quantize_weight
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver


class TestUtils(TestCase):
    # [CLASS DOC] Test case for quantization utility functions; has one shared helper that
    # checks recorded example input ranks and several independent tests.
    def _test_get_fqn_to_example_inputs(self, M, example_inputs, expected_fqn_to_dim):
        # [FUNCTION DOC] HELPER: Instantiate the given module class in eval mode, run
        # get_fqn_to_example_inputs with the example inputs, then for each path in the
        # expected mapping compare the number of dimensions of each recorded input against the
        # expected tuple; raises AssertionError on a mismatch.
        # KNOWN ISSUE: the membership check tests the loop key against the same mapping being
        # iterated, so it can never fail; a missing path would surface as a KeyError instead.
        # Also zip stops at the shorter sequence, so extra recorded inputs are not checked.
        m = M().eval()
        fqn_to_example_inputs = get_fqn_to_example_inputs(m, example_inputs)
        for fqn, expected_dims in expected_fqn_to_dim.items():
            if fqn not in expected_fqn_to_dim:
                raise AssertionError(f"Expected fqn {fqn} in expected_fqn_to_dim")
            example_inputs = fqn_to_example_inputs[fqn]
            for example_input, expected_dim in zip(example_inputs, expected_dims):
                if example_input.dim() != expected_dim:
                    raise AssertionError(
                        f"Expected dim {expected_dim}, got {example_input.dim()}"
                    )

    def test_get_fqn_to_example_inputs_simple(self):
        # [FUNCTION DOC] Builds a parent module (two linears plus a child holding two
        # linears), calls the shared helper with one two-dimensional input and expects every
        # path, from the root through sub.linear2, to have a single two-dimensional recorded
        # input.
        class Sub(torch.nn.Module):
            # [CLASS DOC] HELPER: Child module with two five by five linear layers applied in
            # sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the two linear layers linear1 and linear2.
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply linear1 then linear2 to the input and return
                # the result.
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module with two linears and a Sub child; forward runs
            # the linears and then the child.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create linear1, linear2 and the Sub child.
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                self.sub = Sub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply linear1, linear2 and then the sub child,
                # returning the result.
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sub(x)
                return x

        expected_fqn_to_dim = {
            "": (2,),
            "linear1": (2,),
            "linear2": (2,),
            "sub": (2,),
            "sub.linear1": (2,),
            "sub.linear2": (2,)
        }
        example_inputs = (torch.rand(1, 5),)
        self._test_get_fqn_to_example_inputs(M, example_inputs, expected_fqn_to_dim)

    def test_get_fqn_to_example_inputs_default_kwargs(self):
        # [FUNCTION DOC] Checks that arguments left at their defaults are recorded in the
        # example inputs of a submodule. The child forward takes two defaulted keyword
        # arguments; the parent overrides only the second with a one by two tensor, so the sub
        # path must record dims two, one (default tensor), two (override). Delegates the
        # dimension check to the helper.
        """ Test that we can get example inputs for functions with default keyword arguments
        """
        class Sub(torch.nn.Module):
            # [CLASS DOC] HELPER: Child with two linears whose forward takes the input plus
            # two keyword arguments with default random one-element tensors; the extra
            # arguments are unused.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the two linear layers linear1 and linear2.
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)

            def forward(self, x, key1=torch.rand(1), key2=torch.rand(1)):
                # [FUNCTION DOC] HELPER: Run linear1 and linear2 on x; key1 and key2 are
                # accepted but ignored.
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module whose forward calls the child with only key2
            # overridden.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create linear1, linear2 and the Sub child.
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                self.sub = Sub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply both linears, then call the child with a new
                # random one by two tensor as key2, leaving key1 at its default; returns the
                # result.
                x = self.linear1(x)
                x = self.linear2(x)
                # only override `key2`, `key1` will use default
                x = self.sub(x, key2=torch.rand(1, 2))
                return x

        expected_fqn_to_dim = {
            "": (2,),
            "linear1": (2,),
            "linear2": (2,),
            # second arg is `key1`, which is using default argument
            # third arg is `key2`, override by callsite
            "sub": (2, 1, 2),
            "sub.linear1": (2,),
            "sub.linear2": (2,)
        }
        example_inputs = (torch.rand(1, 5),)
        self._test_get_fqn_to_example_inputs(M, example_inputs, expected_fqn_to_dim)

    def test_get_fqn_to_example_inputs_complex_args(self):
        # [FUNCTION DOC] Checks that non-tensor example inputs are recorded: the child
        # receives a list and a dict, and the recorded inputs for the sub path must contain a
        # list at position one and a dict, with the expected key, at position two. Raises
        # AssertionError otherwise.
        """ Test that we can record complex example inputs such as lists and dicts
        """
        class Sub(torch.nn.Module):
            # [CLASS DOC] HELPER: Child with two linears whose forward also takes a list
            # argument and a dict argument, both ignored.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the two linear layers linear1 and linear2.
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)

            def forward(self, x, list_arg, dict_arg):
                # [FUNCTION DOC] HELPER: Run linear1 and linear2 on x; list_arg and dict_arg
                # are ignored.
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module that calls the child with the activation
            # wrapped in a list and in a dict.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create linear1, linear2 and the Sub child.
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                self.sub = Sub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply both linears, then call the child passing the
                # activation, a one-item list holding it and a dict mapping the key 3 to it;
                # returns the result.
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sub(x, [x], {"3": x})
                return x

        example_inputs = (torch.rand(1, 5),)
        m = M().eval()
        fqn_to_example_inputs = get_fqn_to_example_inputs(m, example_inputs)
        if "sub" not in fqn_to_example_inputs:
            raise AssertionError("Expected 'sub' in fqn_to_example_inputs")
        if not isinstance(fqn_to_example_inputs["sub"][1], list):
            raise AssertionError("Expected fqn_to_example_inputs['sub'][1] to be list")
        if not isinstance(fqn_to_example_inputs["sub"][2], dict):
            raise AssertionError("Expected fqn_to_example_inputs['sub'][2] to be dict")
        if "3" not in fqn_to_example_inputs["sub"][2]:
            raise AssertionError("Expected '3' in fqn_to_example_inputs['sub'][2]")

    def test_quantize_weight_clamping_per_tensor(self):
        # [FUNCTION DOC] Uses a moving-average min/max observer (qint8, symmetric, quant range
        # minus ten to ten, averaging constant one) on a two-element weight of minus/plus one
        # thousand. Asserts the observer min and max equal those values, then that quantizing
        # the weight with _quantize_weight yields integer representation extremes exactly at
        # the observer quant range. Repeats after scaling the weight by 1.2 (values beyond the
        # observed range) and expects the same clamped extremes. Raises AssertionError on
        # mismatch.
        """ Test quant_{min, max} from per tensor observer is honored by `_quantize_weight` method
        """
        fp_min, fp_max = -1000.0, 1000.0
        q8_min, q8_max = -10, 10

        float_tensor = torch.tensor([fp_min, fp_max])

        observer = MovingAverageMinMaxObserver(
            averaging_constant=1.0,
            dtype=torch.qint8,
            quant_min=q8_min,
            quant_max=q8_max,
            qscheme=torch.per_tensor_symmetric,
        )

        observer(float_tensor)
        if observer.min_val != fp_min:
            raise AssertionError(f"Expected min_val {fp_min}, got {observer.min_val}")
        if observer.max_val != fp_max:
            raise AssertionError(f"Expected max_val {fp_max}, got {observer.max_val}")

        quantized_tensor = _quantize_weight(float_tensor, observer)
        if quantized_tensor.int_repr().max().item() != q8_max:
            raise AssertionError(
                f"Expected max {q8_max}, got {quantized_tensor.int_repr().max().item()}"
            )
        if quantized_tensor.int_repr().min().item() != q8_min:
            raise AssertionError(
                f"Expected min {q8_min}, got {quantized_tensor.int_repr().min().item()}"
            )

        # Actual weight values can be outside than observer [min_val, max_val] for the moving average observer
        float_tensor *= 1.2

        quantized_tensor = _quantize_weight(float_tensor, observer)
        if quantized_tensor.int_repr().max().item() != q8_max:
            raise AssertionError(
                f"Expected max {q8_max}, got {quantized_tensor.int_repr().max().item()}"
            )
        if quantized_tensor.int_repr().min().item() != q8_min:
            raise AssertionError(
                f"Expected min {q8_min}, got {quantized_tensor.int_repr().min().item()}"
            )

    def test_quantize_weight_clamping_per_channel(self):
        # [FUNCTION DOC] Per-channel counterpart of the per-tensor clamping test: a moving-
        # average per-channel observer (axis zero, symmetric, quant range minus ten to ten)
        # sees a one by two weight; checks the recorded min and max, then that
        # _quantize_weight clamps the integer representation to the observer quant range, both
        # before and after scaling the weight by 1.2. Raises AssertionError on mismatch.
        """ Test quant_{min, max} from per channel observer is honored by `_quantize_weight` method
        """
        fp_min, fp_max = -1000.0, 1000.0
        q8_min, q8_max = -10, 10

        float_tensor = torch.tensor([[fp_min, fp_max]])

        observer = MovingAveragePerChannelMinMaxObserver(
            averaging_constant=1.0,
            dtype=torch.qint8,
            quant_min=q8_min,
            quant_max=q8_max,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        observer(float_tensor)
        if observer.min_val != fp_min:
            raise AssertionError(f"Expected min_val {fp_min}, got {observer.min_val}")
        if observer.max_val != fp_max:
            raise AssertionError(f"Expected max_val {fp_max}, got {observer.max_val}")

        quantized_tensor = _quantize_weight(float_tensor, observer)
        if quantized_tensor.int_repr().max().item() != q8_max:
            raise AssertionError(
                f"Expected max {q8_max}, got {quantized_tensor.int_repr().max().item()}"
            )
        if quantized_tensor.int_repr().min().item() != q8_min:
            raise AssertionError(
                f"Expected min {q8_min}, got {quantized_tensor.int_repr().min().item()}"
            )

        # Actual weight values can be outside than observer [min_val, max_val] for the moving average observer
        float_tensor *= 1.2

        quantized_tensor = _quantize_weight(float_tensor, observer)
        if quantized_tensor.int_repr().max().item() != q8_max:
            raise AssertionError(
                f"Expected max {q8_max}, got {quantized_tensor.int_repr().max().item()}"
            )
        if quantized_tensor.int_repr().min().item() != q8_min:
            raise AssertionError(
                f"Expected min {q8_min}, got {quantized_tensor.int_repr().min().item()}"
            )

    def test_uint4_int4_dtype(self):
        # [FUNCTION DOC] For each of the uint4 and int4 dtypes, defines a wrapper tensor
        # subclass around a uint8 storage tensor, builds one from a three by eight byte tensor
        # and asserts its dtype equals the sub-byte dtype. The subclass dispatch handler does
        # nothing, so only construction is exercised.

        def up_size(size):
            # [FUNCTION DOC] HELPER: Return a new shape tuple with the last dimension doubled,
            # since each byte packs two four-bit values.
            return (*size[:-1], size[-1] * 2)

        for dtype in [torch.uint4, torch.int4]:
            class UInt4OrInt4Tensor(torch.Tensor):
                # [CLASS DOC] HELPER: Tensor wrapper subclass that reports the enclosing loop
                # dtype with a doubled last dimension and keeps the packed uint8 tensor in the
                # elem attribute. Its dispatch hook is a stub, so no operation on it computes
                # anything.
                @staticmethod
                def __new__(cls, elem, **kwargs):
                    # [FUNCTION DOC] HELPER: Static constructor; requires a uint8 element
                    # tensor (else raises AssertionError) and requires_grad not set to true,
                    # forces requires_grad off, and returns a wrapper subclass instance with
                    # the doubled-last-dimension shape and the loop dtype.
                    if elem.dtype is not torch.uint8:
                        raise AssertionError(f"Expected dtype uint8, got {elem.dtype}")
                    if kwargs.get("requires_grad", False):
                        raise AssertionError("Expected requires_grad to be False")
                    kwargs["requires_grad"] = False
                    return torch.Tensor._make_wrapper_subclass(cls, up_size(elem.shape), dtype=dtype, **kwargs)

                def __init__(self, elem):
                    # [FUNCTION DOC] HELPER: Store the packed uint8 tensor as the elem
                    # attribute.
                    self.elem = elem

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs=None):
                    # [FUNCTION DOC] HELPER: Stub dispatch handler that ignores its arguments
                    # and returns None.
                    pass

            # make sure it runs
            x = UInt4OrInt4Tensor(torch.tensor([
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            ], dtype=torch.uint8))
            if x.dtype != dtype:
                raise AssertionError(f"Expected dtype {dtype}, got {x.dtype}")

if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
