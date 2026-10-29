# [MODULE DOC] Fuzzer operators for tensor elementwise math. Provides stride and broadcast-shape
# helpers, a base class for binary pointwise ops (add, mul, sub, div subclasses) that uses type
# promotion tables to choose input dtypes and may swap an input for a scalar or a broadcast
# shape, plus clamp and cumsum operators. No import-time side effects; operators are found by
# registry introspection.
"""Tensor pointwise operator implementation."""

import random

import torch

from torchfuzz.operators._dtypes import FLOAT_DTYPES
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec
from torchfuzz.type_promotion import (
    get_dtype_map,
    get_dtype_name,
    get_promotion_table_for_strings,
)


def _contiguous_stride(size: tuple[int, ...]) -> tuple[int, ...]:
    # [FUNCTION DOC] HELPER: Return the row-major contiguous stride tuple for a size tuple; an
    # empty size gives an empty tuple. Pure.
    if not size:
        return ()
    strides: list[int] = [1]
    for dim in reversed(size[1:]):
        strides.append(strides[-1] * dim)
    return tuple(reversed(strides))


def _random_broadcast_shape(output_size: tuple[int, ...]) -> tuple[int, ...]:
    # [FUNCTION DOC] HELPER: Return a shape that broadcasts to output_size. Each dimension is
    # independently replaced by 1 with probability 0.3; if none changed, the original size is
    # returned. Otherwise leading size-1 dimensions are dropped one at a time with probability
    # 0.5 each while more than one dimension remains. An empty size returns an empty tuple.
    # Consumes global random state.
    if not output_size:
        return ()
    result = list(output_size)
    changed = False
    for i in range(len(result)):
        if random.random() < 0.3:
            result[i] = 1
            changed = True
    if not changed:
        return output_size
    while len(result) > 1 and result[0] == 1 and random.random() < 0.5:
        result.pop(0)
    return tuple(result)


class PointwiseOperatorBase(Operator):
    # [CLASS DOC] Base for binary elementwise tensor operators. Attributes: symbol, the infix text
    # used when chaining more than two inputs, and the class-level tuple of input positions that
    # may be replaced by a scalar. The registry skips it because its name ends in Base.
    """Base class for element-wise pointwise operations."""

    _scalar_input_positions: tuple[int, ...] = (0, 1)

    def __init__(self, name: str, symbol: str = ""):
        # [FUNCTION DOC] Register under the given name and remember the infix symbol.
        super().__init__(name)
        self.symbol = symbol

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, used as the torch op name.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True for tensor specs whose dtype is not bool; scalars and bool
        # tensors are rejected. Pure check.
        """Tensor pointwise operations can produce tensors but not scalars."""
        if isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool:
            return False
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        # [FUNCTION DOC] Build num_inputs tensor specs with the output size and stride. With two
        # or more inputs, the first two dtypes come from a random promotion pair for the output
        # dtype and the rest use the output dtype; with fewer, all use the output dtype. Then,
        # for two or more inputs, with probability 0.3 one input among the allowed scalar
        # positions becomes a scalar spec, else with probability 0.2 one of the first two becomes
        # a contiguous tensor of a broadcastable smaller shape. Raises ValueError for a
        # non-tensor output spec. Consumes global random state.
        """Decompose tensor into input tensors for pointwise operation with type promotion."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )

        # Use shared type promotion table
        promotion_table = get_promotion_table_for_strings()

        # If num_inputs > 2, promote left-to-right (e.g. (((a op b) op c) op d))
        # For simplicity, we generate the first two with promotion, rest match output dtype
        dtype_str = get_dtype_name(output_spec.dtype)
        supported_types = promotion_table.get(dtype_str, [(dtype_str, dtype_str)])

        # Pick a random promotion pattern for the first two inputs
        if num_inputs >= 2:
            dtypes = list(random.choice(supported_types))
            # For >2 inputs, fill with output dtype
            while len(dtypes) < num_inputs:
                dtypes.append(dtype_str)
        else:
            dtypes = [dtype_str] * num_inputs

        # Convert dtype strings back to torch dtypes
        dtype_map = get_dtype_map()

        input_specs: list[Spec] = [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=dtype_map.get(dt, output_spec.dtype),
            )
            for dt in dtypes
        ]

        if num_inputs >= 2:
            r = random.random()
            if self._scalar_input_positions and r < 0.3:
                idx = random.choice(self._scalar_input_positions)
                input_specs[idx] = ScalarSpec(dtype=input_specs[idx].dtype)
            elif r < 0.5:
                idx = random.randint(0, 1)
                bcast_size = _random_broadcast_shape(tuple(output_spec.size))
                if bcast_size != tuple(output_spec.size):
                    input_specs[idx] = TensorSpec(
                        size=bcast_size,
                        stride=_contiguous_stride(bcast_size),
                        dtype=input_specs[idx].dtype,
                    )

        return input_specs

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return source assigning output_name. With exactly two inputs it calls
        # the torch op on them; otherwise it joins all input names with the infix symbol.
        """Generate code for pointwise operation."""
        if len(input_names) == 2:
            return f"{output_name} = {self.torch_op_name}({input_names[0]}, {input_names[1]})"
        else:
            # Chain operations using symbols for readability
            expr = f" {self.symbol} ".join(input_names)
            return f"{output_name} = {expr}"


class AddOperator(PointwiseOperatorBase):
    # [CLASS DOC] Elementwise addition operator, name add. Sets its selection weight from the
    # constructor argument. Codegen sometimes adds a random alpha scale factor.
    """Operator for element-wise addition."""

    def __init__(self, weight: float = 1.0):
        # [FUNCTION DOC] Register as add with the plus symbol and store the weight as a float.
        super().__init__("add", "+")
        self.weight = float(weight)

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the string torch.add.
        return "torch.add"

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] With two inputs and probability 0.3, return source calling add with an
        # alpha keyword drawn uniformly in the range from -5 to 5 rounded to four decimals,
        # truncated to an integer unless the output dtype is a float type. Otherwise defer to the
        # base class. Consumes global random state.
        # KNOWN ISSUE: a non-tensor output spec is treated as float, so alpha stays fractional.
        if len(input_names) == 2 and random.random() < 0.3:
            alpha = round(random.uniform(-5.0, 5.0), 4)
            if isinstance(output_spec, TensorSpec) and output_spec.dtype not in (
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ):
                alpha = int(alpha)
            return (
                f"{output_name} = torch.add("
                f"{input_names[0]}, {input_names[1]}, alpha={alpha!r})"
            )
        return super().codegen(output_name, input_names, output_spec)


class MulOperator(PointwiseOperatorBase):
    # [CLASS DOC] Elementwise multiplication operator, name mul, symbol star.
    """Operator for element-wise multiplication."""

    def __init__(self):
        # [FUNCTION DOC] Register as mul with the multiplication symbol.
        super().__init__("mul", "*")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the string torch.mul.
        return "torch.mul"


class SubOperator(PointwiseOperatorBase):
    # [CLASS DOC] Elementwise subtraction operator, name sub, symbol minus.
    """Operator for element-wise subtraction."""

    def __init__(self):
        # [FUNCTION DOC] Register as sub with the minus symbol.
        super().__init__("sub", "-")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the string torch.sub.
        return "torch.sub"


class DivOperator(PointwiseOperatorBase):
    # [CLASS DOC] Elementwise division operator, name div, symbol slash. No zero guard is applied.
    """Operator for element-wise division."""

    def __init__(self):
        # [FUNCTION DOC] Register as div with the division symbol.
        super().__init__("div", "/")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the string torch.div.
        return "torch.div"


class ClampOperator(Operator):
    # [CLASS DOC] Operator for torch.clamp, name clamp. Stateless; takes one input matching the
    # output and emits constant bounds of -1 and 1 in float or integer form by dtype.
    """Operator for torch.clamp (element-wise clamping)."""

    def __init__(self):
        # [FUNCTION DOC] Register under the name clamp.
        super().__init__("clamp")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the string torch.clamp.
        return "torch.clamp"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True for tensor specs whose dtype is not bool. Pure check.
        """Clamp can produce tensors but not scalars."""
        if isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool:
            return False
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Return one tensor spec copying the output size, stride and dtype. Raises
        # ValueError if output_spec is not a tensor spec. The min and max bounds are not inputs.
        """Generate input specs for clamp operation.

        Clamp takes:
        - input tensor (same shape and dtype as output)
        - optional min value (scalar or None)
        - optional max value (scalar or None)
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ClampOperator can only produce TensorSpec outputs")

        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return source calling clamp on the single input with min and max keywords.
        # Each bound is used with probability about 0.67, independently, and the min is forced on
        # if both were skipped; a skipped bound is written as None. Bounds are -1.0 and 1.0 for
        # float dtypes, -1 and 1 otherwise. Raises ValueError unless exactly one input name is
        # given and AssertionError for a non-tensor spec. Consumes global random state.
        """Generate code for clamp operation."""
        if len(input_names) != 1:
            raise ValueError("Clamp requires exactly 1 input tensor")

        input_name = input_names[0]

        # Generate random min and max values for clamping
        # We'll randomly decide whether to use min, max, or both
        use_min = random.random() > 0.33
        use_max = random.random() > 0.33

        if not use_min and not use_max:
            use_min = True

        # For integer-dtype tensors, emit integer bounds so the fuzzer doesn't
        # generate float-bound + int-tensor mismatches between CPU and MTIA.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        if output_spec.dtype in FLOAT_DTYPES:
            min_literal, max_literal = "-1.0", "1.0"
        else:
            min_literal, max_literal = "-1", "1"

        args = [input_name]
        if use_min:
            args.append(f"min={min_literal}")
        else:
            args.append("min=None")

        if use_max:
            args.append(f"max={max_literal}")
        else:
            args.append("max=None")

        return f"{output_name} = torch.clamp({', '.join(args)})"


class CumsumOperator(Operator):
    # [CLASS DOC] Operator for torch.cumsum, name cumsum. Stateless; input matches the output
    # spec and the accumulation dimension is chosen randomly at code generation time.
    """Operator for torch.cumsum (cumulative sum along a dimension)."""

    def __init__(self):
        # [FUNCTION DOC] Register under the name cumsum.
        super().__init__("cumsum")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the string torch.cumsum.
        return "torch.cumsum"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True for non-bool tensor specs with at least one dimension.
        """Cumsum can produce tensors but not scalars."""
        if isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool:
            return False
        # Cumsum needs at least 1 dimension
        if isinstance(output_spec, TensorSpec) and len(output_spec.size) == 0:
            return False
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Return one tensor spec copying the output size, stride and dtype. Raises
        # ValueError if output_spec is not a tensor spec.
        """Generate input specs for cumsum operation.

        Cumsum takes an input tensor with same shape and dtype as output.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("CumsumOperator can only produce TensorSpec outputs")

        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return source calling cumsum on the single input along a dimension drawn
        # uniformly from the output's dimensions. Raises ValueError unless exactly one input name
        # is given, the output is a tensor spec, and it has at least one dimension. Consumes
        # global random state.
        """Generate code for cumsum operation."""
        if len(input_names) != 1:
            raise ValueError("Cumsum requires exactly 1 input tensor")

        if not isinstance(output_spec, TensorSpec):
            raise ValueError("Output spec must be a TensorSpec")

        input_name = input_names[0]

        # Choose a random valid dimension
        num_dims = len(output_spec.size)
        if num_dims == 0:
            raise ValueError("Cumsum requires tensor with at least 1 dimension")

        # Pick a random dimension index
        dim = random.randint(0, num_dims - 1)

        return f"{output_name} = torch.cumsum({input_name}, dim={dim})"
