# [MODULE DOC] Fuzzer operators for torch.nn.functional ops and activations: embedding, linear,
# relu, softmax, dropout, layer_norm, rms_norm, gelu, sigmoid, tanh, batch_norm, group_norm,
# leaky_relu, elu, silu, scaled_dot_product_attention and multi_head_attention_forward. Also
# defines a float-dtype predicate. Most operators accept only float dtypes, and their codegen
# casts inputs to the output dtype. No import-time side effects.
"""Neural network functional operator implementations."""

import math
import random

import torch

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


def is_float_dtype(dtype: torch.dtype) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if the dtype is float16, bfloat16, float32
    # or float64.
    """Check if dtype is a floating point type."""
    return dtype in [
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
    ]


class EmbeddingOperator(Operator):
    # [CLASS DOC] Operator for embedding lookup: a weight table plus integer indices. Stateless.
    """Operator for torch.nn.functional.embedding."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the
        # name torch.nn.functional.embedding.
        super().__init__("torch.nn.functional.embedding")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.embedding.
        """Return the torch operation name."""
        return "torch.nn.functional.embedding"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs with at least one dimension and a
        # float dtype.
        """Embedding can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Embedding needs at least 1 dimension (embedding_dim)
        if len(output_spec.size) == 0:
            return False
        # Embedding outputs are typically float tensors
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Treats the last output dimension as the embedding size and the rest as
        # the index shape. Returns a weight spec (random vocabulary of 150 to 500 rows by
        # embedding size, output dtype) and an int64 index spec with a contiguous stride. Raises
        # ValueError for non-tensor or 0-d specs. Uses the random module.
        """Generate input specs for embedding operation.

        Embedding requires:
        - weight tensor: (num_embeddings, embedding_dim)
        - input tensor: integer indices (any shape, but output shape + [embedding_dim])
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("EmbeddingOperator can only produce TensorSpec outputs")

        # Output shape should be input_shape + [embedding_dim]
        if len(output_spec.size) == 0:
            raise ValueError("Embedding output must have at least 1 dimension")

        embedding_dim = output_spec.size[-1]
        input_shape = output_spec.size[:-1]  # Remove last dimension for embedding_dim

        # Generate reasonable vocab size that's larger than our index generation range
        # This ensures that indices generated in range [0, 100) will always be valid
        num_embeddings = random.randint(150, 500)  # Always larger than max index (100)

        # Weight tensor: (num_embeddings, embedding_dim)
        weight_spec = TensorSpec(
            size=(num_embeddings, embedding_dim),
            stride=(embedding_dim, 1),
            dtype=output_spec.dtype,
        )

        # Input tensor: integer indices with shape that produces the output shape
        input_spec = TensorSpec(
            size=input_shape,
            stride=self._calculate_stride(input_shape),
            dtype=torch.int64,  # Indices are typically int64
        )

        return [weight_spec, input_spec]

    def _calculate_stride(self, size):
        # [FUNCTION DOC] HELPER: Returns contiguous strides for a size (empty tuple for an
        # empty size).
        """Calculate stride for a given size."""
        if not size:
            return ()
        stride = []
        current_stride = 1
        for dim_size in reversed(size):
            stride.append(current_stride)
            current_stride *= dim_size
        return tuple(reversed(stride))

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line calling embedding with the index input converted to int64
        # and clamped into the valid row range of the weight, so out-of-range values are safe.
        # Input order is weight then indices. Raises ValueError unless there are exactly two
        # input names.
        """Generate code for embedding operation."""
        if len(input_names) != 2:
            raise ValueError("Embedding requires exactly 2 inputs: weight and input")

        weight_name, input_name = input_names
        # Ensure indices are integer type and clamped to valid range
        # This handles any arithmetic operations that might produce out-of-bounds indices
        return f"{output_name} = torch.nn.functional.embedding(torch.clamp({input_name}.to(torch.int64), 0, {weight_name}.size(0)-1), {weight_name})"


class LinearOperator(Operator):
    # [CLASS DOC] Operator for a linear layer with optional bias. Stateless.
    """Operator for torch.nn.functional.linear."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the
        # name torch.nn.functional.linear.
        super().__init__("torch.nn.functional.linear")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.linear.
        """Return the torch operation name."""
        return "torch.nn.functional.linear"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs with at least one dimension and a
        # float dtype.
        """Linear can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Linear needs at least 1 dimension (output features)
        if len(output_spec.size) == 0:
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Treats the last output dimension as out_features. Picks a random
        # in_features from 8 to 256 and returns an input spec (leading dims plus in_features,
        # contiguous), a weight spec (out_features by in_features) and, with probability 0.5, a
        # bias spec. Dtype is the output dtype. Raises ValueError for non-tensor or 0-d specs.
        # Uses the random module.
        """Generate input specs for linear operation.

        Linear transformation: y = xW^T + b
        - input: (..., in_features)
        - weight: (out_features, in_features)
        - bias: (out_features,) [optional]
        - output: (..., out_features)
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("LinearOperator can only produce TensorSpec outputs")

        if len(output_spec.size) == 0:
            raise ValueError("Linear output must have at least 1 dimension")

        out_features = output_spec.size[-1]
        batch_shape = output_spec.size[:-1]

        # Generate reasonable input features size
        in_features = random.randint(8, 256)

        # Input tensor: (..., in_features)
        input_shape = batch_shape + (in_features,)
        input_spec = TensorSpec(
            size=input_shape,
            stride=self._calculate_stride(input_shape),
            dtype=output_spec.dtype,
        )

        # Weight tensor: (out_features, in_features)
        weight_spec = TensorSpec(
            size=(out_features, in_features),
            stride=(in_features, 1),
            dtype=output_spec.dtype,
        )

        # Bias tensor: (out_features,) - make bias optional with 50% probability
        if random.random() < 0.5:
            bias_spec = TensorSpec(
                size=(out_features,), stride=(1,), dtype=output_spec.dtype
            )
            return [input_spec, weight_spec, bias_spec]
        else:
            return [input_spec, weight_spec]

    def _calculate_stride(self, size):
        # [FUNCTION DOC] HELPER: Returns contiguous strides for a size (empty tuple for an
        # empty size).
        """Calculate stride for a given size."""
        if not size:
            return ()
        stride = []
        current_stride = 1
        for dim_size in reversed(size):
            stride.append(current_stride)
            current_stride *= dim_size
        return tuple(reversed(stride))

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line calling linear with each input cast to the output dtype;
        # takes two names (input, weight) or three (plus bias). Raises ValueError for any other
        # count or a non-TensorSpec output.
        """Generate code for linear operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("LinearOperator can only produce TensorSpec outputs")

        # Ensure dtype compatibility by converting all inputs to the expected output dtype
        target_dtype = str(output_spec.dtype)

        if len(input_names) == 2:
            input_name, weight_name = input_names
            return f"{output_name} = torch.nn.functional.linear({input_name}.to({target_dtype}), {weight_name}.to({target_dtype}))"
        elif len(input_names) == 3:
            input_name, weight_name, bias_name = input_names
            return f"{output_name} = torch.nn.functional.linear({input_name}.to({target_dtype}), {weight_name}.to({target_dtype}), {bias_name}.to({target_dtype}))"
        else:
            raise ValueError(
                "Linear requires 2 or 3 inputs: input, weight, and optional bias"
            )


class ReLUOperator(Operator):
    # [CLASS DOC] Elementwise relu operator: one input spec identical to the output
    # spec. Stateless.
    """Operator for torch.nn.functional.relu."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the name torch.nn.functional.relu.
        super().__init__("torch.nn.functional.relu")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.relu.
        """Return the torch operation name."""
        return "torch.nn.functional.relu"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs with float dtype (float16, bfloat16,
        # float32, float64).
        """ReLU can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec with the same size, stride and dtype as the
        # output. Raises ValueError if output_spec is not a TensorSpec.
        """Generate input specs for ReLU operation.

        ReLU is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ReLUOperator can only produce TensorSpec outputs")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line applying relu to the single input. Raises ValueError if
        # the number of input names is not exactly one.
        """Generate code for ReLU operation."""
        if len(input_names) != 1:
            raise ValueError("ReLU requires exactly 1 input")

        input_name = input_names[0]
        return f"{output_name} = torch.nn.functional.relu({input_name})"


class SoftmaxOperator(Operator):
    # [CLASS DOC] Elementwise softmax operator. Stateless.
    """Operator for torch.nn.functional.softmax."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the
        # name torch.nn.functional.softmax.
        super().__init__("torch.nn.functional.softmax")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.softmax.
        """Return the torch operation name."""
        return "torch.nn.functional.softmax"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for float tensor specs with at least one dimension.
        """Softmax can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # Softmax needs at least 1 dimension to apply softmax along a dimension
        if len(output_spec.size) == 0:
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec with the same size, stride and dtype as the
        # output. Raises ValueError if output_spec is not a TensorSpec.
        """Generate input specs for softmax operation.

        Softmax is element-wise along a dimension, input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("SoftmaxOperator can only produce TensorSpec outputs")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line applying softmax to the single input along the last
        # dimension. Raises ValueError if the number of input names is wrong.
        """Generate code for softmax operation."""
        if len(input_names) != 1:
            raise ValueError("Softmax requires exactly 1 input")

        input_name = input_names[0]
        # Use dim=-1 as default (last dimension)
        return f"{output_name} = torch.nn.functional.softmax({input_name}, dim=-1)"


class DropoutOperator(Operator):
    # [CLASS DOC] Elementwise dropout operator. Stateless.
    """Operator for torch.nn.functional.dropout."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the
        # name torch.nn.functional.dropout.
        super().__init__("torch.nn.functional.dropout")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.dropout.
        """Return the torch operation name."""
        return "torch.nn.functional.dropout"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs with float dtype (float16, bfloat16,
        # float32, float64).
        """Dropout can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec with the same size, stride and dtype as the
        # output. Raises ValueError if output_spec is not a TensorSpec.
        """Generate input specs for dropout operation.

        Dropout is element-wise, input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("DropoutOperator can only produce TensorSpec outputs")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line applying dropout to the single input with probability 0.1
        # and training disabled, making it deterministic (an identity). Raises ValueError if the
        # number of input names is wrong.
        """Generate code for dropout operation."""
        if len(input_names) != 1:
            raise ValueError("Dropout requires exactly 1 input")

        input_name = input_names[0]
        # Use training=False to make it deterministic for testing
        return f"{output_name} = torch.nn.functional.dropout({input_name}, p=0.1, training=False)"


class LayerNormOperator(Operator):
    # [CLASS DOC] Operator for layer normalization over the last dimension, with optional weight
    # and bias. Stateless.
    """Operator for torch.nn.functional.layer_norm."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the
        # name torch.nn.functional.layer_norm.
        super().__init__("torch.nn.functional.layer_norm")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.layer_norm.
        """Return the torch operation name."""
        return "torch.nn.functional.layer_norm"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for float tensor specs with at least one dimension.
        """LayerNorm can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # LayerNorm needs at least 1 dimension to normalize over
        if len(output_spec.size) == 0:
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns an input spec equal to the output spec, then with probability
        # 0.7 a weight spec over the last dimension and, only if a weight was added, with
        # probability 0.7 a bias spec; all in the output dtype with stride one. Raises
        # ValueError for non-tensor or 0-d specs. Uses the random module.
        """Generate input specs for layer_norm operation.

        LayerNorm normalizes over the last dimensions specified by normalized_shape.
        - input: input tensor
        - weight: (normalized_shape,) [optional]
        - bias: (normalized_shape,) [optional]
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("LayerNormOperator can only produce TensorSpec outputs")

        if len(output_spec.size) == 0:
            raise ValueError("LayerNorm output must have at least 1 dimension")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        # For simplicity, normalize over the last dimension
        normalized_shape = output_spec.size[-1:]

        # Weight and bias tensors (optional with 70% probability each)
        specs = [input_spec]
        if random.random() < 0.7:
            # LayerNorm weight and bias parameters should match input tensor dtype
            # for compatibility (conversion will be handled in codegen)
            weight_spec = TensorSpec(
                size=normalized_shape, stride=(1,), dtype=output_spec.dtype
            )
            specs.append(weight_spec)

            if random.random() < 0.7:
                bias_spec = TensorSpec(
                    size=normalized_shape, stride=(1,), dtype=output_spec.dtype
                )
                specs.append(bias_spec)

        # Cast to list[Spec] to fix type checking
        from typing import cast

        return cast(list[Spec], specs)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line calling layer_norm over the last dimension with inputs
        # cast to the output dtype; one, two or three names mean input, input with weight,
        # or input with weight and bias. Raises ValueError for another count or a non-
        # TensorSpec output.
        """Generate code for layer_norm operation."""
        if len(input_names) < 1 or len(input_names) > 3:
            raise ValueError(
                "LayerNorm requires 1-3 inputs: input, optional weight, optional bias"
            )

        if not isinstance(output_spec, TensorSpec):
            raise ValueError("LayerNormOperator can only produce TensorSpec outputs")

        # Normalize over the last dimension
        normalized_shape = f"({output_spec.size[-1]},)"

        # Ensure dtype compatibility by converting all inputs to the expected output dtype
        target_dtype = str(output_spec.dtype)

        input_name = input_names[0]

        if len(input_names) == 1:
            return f"{output_name} = torch.nn.functional.layer_norm({input_name}.to({target_dtype}), {normalized_shape})"
        elif len(input_names) == 2:
            weight_name = input_names[1]
            return f"{output_name} = torch.nn.functional.layer_norm({input_name}.to({target_dtype}), {normalized_shape}, weight={weight_name}.to({target_dtype}))"
        else:  # len(input_names) == 3
            weight_name, bias_name = input_names[1], input_names[2]
            return f"{output_name} = torch.nn.functional.layer_norm({input_name}.to({target_dtype}), {normalized_shape}, weight={weight_name}.to({target_dtype}), bias={bias_name}.to({target_dtype}))"


class RMSNormOperator(Operator):
    # [CLASS DOC] Operator for RMS normalization over the last dimension with optional weight.
    # Sampling weight attribute is 5.0.
    """Operator for torch.nn.functional.rms_norm (Root Mean Square Normalization).

    RMSNorm is commonly used in modern LLMs like LLaMA. It normalizes by the RMS of the input.
    """

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register under the name torch.nn.functional.rms_norm and set
        # the weight attribute to 5.0.
        super().__init__("torch.nn.functional.rms_norm")
        self.weight = 5.0

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.rms_norm.
        """Return the torch operation name."""
        return "torch.nn.functional.rms_norm"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for float tensor specs with at least one dimension.
        """RMSNorm can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # RMSNorm needs at least 1 dimension to normalize over
        if len(output_spec.size) == 0:
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns an input spec equal to the output spec and, with probability
        # 0.7, a weight spec over the last dimension (output dtype, stride one). Raises
        # ValueError for non-tensor or 0-d specs. Uses the random module.
        """Generate input specs for RMSNorm operation.

        RMSNorm requires:
        - input: input tensor
        - weight: (normalized_shape,) [optional]
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("RMSNormOperator can only produce TensorSpec outputs")

        if len(output_spec.size) == 0:
            raise ValueError("RMSNorm output must have at least 1 dimension")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        # Weight tensor (optional with 70% probability)
        normalized_shape = output_spec.size[-1:]
        specs = [input_spec]
        if random.random() < 0.7:
            weight_spec = TensorSpec(
                size=normalized_shape, stride=(1,), dtype=output_spec.dtype
            )
            specs.append(weight_spec)

        from typing import cast

        return cast(list[Spec], specs)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line calling rms_norm over the last dimension with inputs cast
        # to the output dtype; one name is the input, two add a weight. Raises ValueError for
        # another count or a non-TensorSpec output.
        """Generate code for RMSNorm operation."""
        if len(input_names) < 1 or len(input_names) > 2:
            raise ValueError("RMSNorm requires 1-2 inputs: input, optional weight")

        if not isinstance(output_spec, TensorSpec):
            raise ValueError("RMSNormOperator can only produce TensorSpec outputs")

        target_dtype = str(output_spec.dtype)
        input_name = input_names[0]

        # Normalize over the last dimension
        normalized_shape = f"({output_spec.size[-1]},)"

        if len(input_names) == 1:
            return f"{output_name} = torch.nn.functional.rms_norm({input_name}.to({target_dtype}), {normalized_shape})"
        else:  # len(input_names) == 2
            weight_name = input_names[1]
            return f"{output_name} = torch.nn.functional.rms_norm({input_name}.to({target_dtype}), {normalized_shape}, weight={weight_name}.to({target_dtype}))"


class GELUOperator(Operator):
    # [CLASS DOC] Elementwise gelu operator. Stateless.
    """Operator for torch.nn.functional.gelu (Gaussian Error Linear Unit)."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the name torch.nn.functional.gelu.
        super().__init__("torch.nn.functional.gelu")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.gelu.
        """Return the torch operation name."""
        return "torch.nn.functional.gelu"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs with float dtype (float16, bfloat16,
        # float32, float64).
        """GELU can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec with the same size, stride and dtype as the
        # output. Raises ValueError if output_spec is not a TensorSpec.
        """Generate input specs for GELU operation.

        GELU is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("GELUOperator can only produce TensorSpec outputs")

        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line applying gelu to the single input with the approximate
        # mode chosen at random between none and tanh (drawn from the global random module at
        # code generation time). Raises ValueError if the number of input names is wrong.
        """Generate code for GELU operation."""
        if len(input_names) != 1:
            raise ValueError("GELU requires exactly 1 input")

        input_name = input_names[0]
        approx = random.choice(["none", "tanh"])
        return f"{output_name} = torch.nn.functional.gelu({input_name}, approximate={approx!r})"


class SigmoidOperator(Operator):
    # [CLASS DOC] Elementwise sigmoid operator. Stateless.
    """Operator for torch.sigmoid."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the name torch.sigmoid.
        super().__init__("torch.sigmoid")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.sigmoid.
        """Return the torch operation name."""
        return "torch.sigmoid"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs with float dtype (float16, bfloat16,
        # float32, float64).
        """Sigmoid can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec with the same size, stride and dtype as the
        # output. Raises ValueError if output_spec is not a TensorSpec.
        """Generate input specs for sigmoid operation.

        Sigmoid is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("SigmoidOperator can only produce TensorSpec outputs")

        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line applying sigmoid to the single input. Raises ValueError if
        # the number of input names is wrong.
        """Generate code for sigmoid operation."""
        if len(input_names) != 1:
            raise ValueError("Sigmoid requires exactly 1 input")

        input_name = input_names[0]
        return f"{output_name} = torch.sigmoid({input_name})"


class TanhOperator(Operator):
    # [CLASS DOC] Elementwise tanh operator. Stateless.
    """Operator for torch.tanh."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the name torch.tanh.
        super().__init__("torch.tanh")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.tanh.
        """Return the torch operation name."""
        return "torch.tanh"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs with float dtype (float16, bfloat16,
        # float32, float64).
        """Tanh can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec with the same size, stride and dtype as the
        # output. Raises ValueError if output_spec is not a TensorSpec.
        """Generate input specs for tanh operation.

        Tanh is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("TanhOperator can only produce TensorSpec outputs")

        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line applying tanh to the single input. Raises ValueError if
        # the number of input names is wrong.
        """Generate code for tanh operation."""
        if len(input_names) != 1:
            raise ValueError("Tanh requires exactly 1 input")

        input_name = input_names[0]
        return f"{output_name} = torch.tanh({input_name})"


class BatchNormOperator(Operator):
    # [CLASS DOC] Operator for batch normalization in inference mode with running statistics and
    # optional weight and bias. Stateless.
    """Operator for torch.nn.functional.batch_norm."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the
        # name torch.nn.functional.batch_norm.
        super().__init__("torch.nn.functional.batch_norm")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.batch_norm.
        """Return the torch operation name."""
        return "torch.nn.functional.batch_norm"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for float tensor specs with at least two dimensions and a
        # nonzero channel (second) dimension.
        """BatchNorm can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # BatchNorm needs at least 2 dimensions (batch, features)
        if len(output_spec.size) < 2:
            return False
        # Channel dimension (second dimension) must be greater than 0
        if output_spec.size[1] == 0:
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns an input spec equal to the output spec, running mean and
        # running variance specs over the channel dimension, then with probability 0.7 a weight
        # spec and (only after a weight) with probability 0.7 a bias spec; all in the output
        # dtype. Raises ValueError for non-tensor specs or fewer than two dimensions. Uses the
        # random module.
        """Generate input specs for batch_norm operation.

        BatchNorm requires:
        - input: (N, C, ...) where N is batch and C is channels
        - running_mean: (C,) [optional]
        - running_var: (C,) [optional]
        - weight: (C,) [optional]
        - bias: (C,) [optional]
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("BatchNormOperator can only produce TensorSpec outputs")

        if len(output_spec.size) < 2:
            raise ValueError("BatchNorm output must have at least 2 dimensions")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        # Channel dimension is the second dimension
        num_features = output_spec.size[1]

        specs = [input_spec]

        # Add running_mean and running_var (required for inference mode)
        running_mean_spec = TensorSpec(
            size=(num_features,), stride=(1,), dtype=output_spec.dtype
        )
        running_var_spec = TensorSpec(
            size=(num_features,), stride=(1,), dtype=output_spec.dtype
        )
        specs.extend([running_mean_spec, running_var_spec])

        # Add weight and bias (optional with 70% probability)
        if random.random() < 0.7:
            weight_spec = TensorSpec(
                size=(num_features,), stride=(1,), dtype=output_spec.dtype
            )
            specs.append(weight_spec)

            if random.random() < 0.7:
                bias_spec = TensorSpec(
                    size=(num_features,), stride=(1,), dtype=output_spec.dtype
                )
                specs.append(bias_spec)

        from typing import cast

        return cast(list[Spec], specs)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line calling batch_norm with training disabled; running mean
        # and variance are detached (they must not require grad) and all inputs are cast to the
        # output dtype. Three to five names mean input, running mean, running variance, optional
        # weight, optional bias. Raises ValueError for another count or a non-TensorSpec output.
        """Generate code for batch_norm operation."""
        if len(input_names) < 3 or len(input_names) > 5:
            raise ValueError(
                "BatchNorm requires 3-5 inputs: input, running_mean, running_var, optional weight, optional bias"
            )

        if not isinstance(output_spec, TensorSpec):
            raise ValueError("BatchNormOperator can only produce TensorSpec outputs")

        target_dtype = str(output_spec.dtype)
        input_name = input_names[0]
        # running_mean/running_var must not require grad (batch_norm is not
        # differentiable w.r.t. them). Args are marked requires_grad by the
        # template, so detach these two here.
        running_mean_name = f"{input_names[1]}.detach()"
        running_var_name = f"{input_names[2]}.detach()"

        # Use training=False for deterministic behavior
        if len(input_names) == 3:
            return f"{output_name} = torch.nn.functional.batch_norm({input_name}.to({target_dtype}), {running_mean_name}.to({target_dtype}), {running_var_name}.to({target_dtype}), training=False)"
        elif len(input_names) == 4:
            weight_name = input_names[3]
            return f"{output_name} = torch.nn.functional.batch_norm({input_name}.to({target_dtype}), {running_mean_name}.to({target_dtype}), {running_var_name}.to({target_dtype}), weight={weight_name}.to({target_dtype}), training=False)"
        else:  # len(input_names) == 5
            weight_name = input_names[3]
            bias_name = input_names[4]
            return f"{output_name} = torch.nn.functional.batch_norm({input_name}.to({target_dtype}), {running_mean_name}.to({target_dtype}), {running_var_name}.to({target_dtype}), weight={weight_name}.to({target_dtype}), bias={bias_name}.to({target_dtype}), training=False)"


class GroupNormOperator(Operator):
    # [CLASS DOC] Operator for group normalization with optional weight and bias. Stateless.
    """Operator for torch.nn.functional.group_norm."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the
        # name torch.nn.functional.group_norm.
        super().__init__("torch.nn.functional.group_norm")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.group_norm.
        """Return the torch operation name."""
        return "torch.nn.functional.group_norm"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for float tensor specs with at least two dimensions where
        # the batch size times the product of the spatial sizes exceeds one.
        """GroupNorm can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # GroupNorm needs at least 2 dimensions (batch, channels)
        if len(output_spec.size) < 2:
            return False

        # GroupNorm requires more than 1 value per channel
        # For shape (N, C, *), num_values_per_channel = N * prod(*)
        # We need N * prod(*) > 1
        batch_size = output_spec.size[0]
        spatial_size = math.prod(output_spec.size[2:])
        num_values_per_channel = batch_size * spatial_size

        if num_values_per_channel <= 1:
            return False

        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns an input spec equal to the output spec, then with probability
        # 0.7 a weight spec over the channel (second) dimension and, only after a weight, with
        # probability 0.7 a bias spec; all in the output dtype. Raises ValueError for non-tensor
        # specs or fewer than two dimensions. Uses the random module.
        """Generate input specs for group_norm operation.

        GroupNorm requires:
        - input: (N, C, ...) where N is batch and C is channels
        - weight: (C,) [optional]
        - bias: (C,) [optional]
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("GroupNormOperator can only produce TensorSpec outputs")

        if len(output_spec.size) < 2:
            raise ValueError("GroupNorm output must have at least 2 dimensions")

        # Input tensor has same shape and dtype as output
        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        # Channel dimension is the second dimension
        num_channels = output_spec.size[1]

        specs = [input_spec]

        # Add weight and bias (optional with 70% probability)
        if random.random() < 0.7:
            weight_spec = TensorSpec(
                size=(num_channels,), stride=(1,), dtype=output_spec.dtype
            )
            specs.append(weight_spec)

            if random.random() < 0.7:
                bias_spec = TensorSpec(
                    size=(num_channels,), stride=(1,), dtype=output_spec.dtype
                )
                specs.append(bias_spec)

        from typing import cast

        return cast(list[Spec], specs)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line calling group_norm with inputs cast to the output dtype
        # and a group count equal to the first of 32, 16, 8, 4, 2, 1 that divides the channel
        # count. One to three names mean input, optional weight, optional bias. Raises
        # ValueError for another count or a non-TensorSpec output.
        # KNOWN ISSUE: the fallback group count branch is unreachable because 1 always divides.
        """Generate code for group_norm operation."""
        if len(input_names) < 1 or len(input_names) > 3:
            raise ValueError(
                "GroupNorm requires 1-3 inputs: input, optional weight, optional bias"
            )

        if not isinstance(output_spec, TensorSpec):
            raise ValueError("GroupNormOperator can only produce TensorSpec outputs")

        target_dtype = str(output_spec.dtype)
        input_name = input_names[0]

        # Determine number of groups (must divide num_channels evenly)
        num_channels = output_spec.size[1]
        # Common choices: 32, 16, 8, or equal to channels (instance norm)
        possible_groups = [g for g in [32, 16, 8, 4, 2, 1] if num_channels % g == 0]
        num_groups = possible_groups[0] if possible_groups else 1

        if len(input_names) == 1:
            return f"{output_name} = torch.nn.functional.group_norm({input_name}.to({target_dtype}), {num_groups})"
        elif len(input_names) == 2:
            weight_name = input_names[1]
            return f"{output_name} = torch.nn.functional.group_norm({input_name}.to({target_dtype}), {num_groups}, weight={weight_name}.to({target_dtype}))"
        else:  # len(input_names) == 3
            weight_name = input_names[1]
            bias_name = input_names[2]
            return f"{output_name} = torch.nn.functional.group_norm({input_name}.to({target_dtype}), {num_groups}, weight={weight_name}.to({target_dtype}), bias={bias_name}.to({target_dtype}))"


class LeakyReLUOperator(Operator):
    # [CLASS DOC] Elementwise leaky_relu operator. Stateless.
    """Operator for torch.nn.functional.leaky_relu."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the
        # name torch.nn.functional.leaky_relu.
        super().__init__("torch.nn.functional.leaky_relu")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.leaky_relu.
        """Return the torch operation name."""
        return "torch.nn.functional.leaky_relu"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs with float dtype (float16, bfloat16,
        # float32, float64).
        """LeakyReLU can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec with the same size, stride and dtype as the
        # output. Raises ValueError if output_spec is not a TensorSpec.
        """Generate input specs for LeakyReLU operation.

        LeakyReLU is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("LeakyReLUOperator can only produce TensorSpec outputs")

        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line applying leaky_relu to the single input with a random
        # negative slope between 0.001 and 0.5 rounded to 4 digits (drawn at code generation
        # time). Raises ValueError if the number of input names is wrong.
        """Generate code for LeakyReLU operation."""
        if len(input_names) != 1:
            raise ValueError("LeakyReLU requires exactly 1 input")

        input_name = input_names[0]
        slope = round(random.uniform(0.001, 0.5), 4)
        return f"{output_name} = torch.nn.functional.leaky_relu({input_name}, negative_slope={slope!r})"


class ELUOperator(Operator):
    # [CLASS DOC] Elementwise elu operator. Stateless.
    """Operator for torch.nn.functional.elu (Exponential Linear Unit)."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the name torch.nn.functional.elu.
        super().__init__("torch.nn.functional.elu")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.elu.
        """Return the torch operation name."""
        return "torch.nn.functional.elu"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs with float dtype (float16, bfloat16,
        # float32, float64).
        """ELU can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec with the same size, stride and dtype as the
        # output. Raises ValueError if output_spec is not a TensorSpec.
        """Generate input specs for ELU operation.

        ELU is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ELUOperator can only produce TensorSpec outputs")

        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line applying elu to the single input with a random alpha
        # between 0.1 and 3.0 rounded to 4 digits (drawn at code generation time). Raises
        # ValueError if the number of input names is wrong.
        """Generate code for ELU operation."""
        if len(input_names) != 1:
            raise ValueError("ELU requires exactly 1 input")

        input_name = input_names[0]
        alpha = round(random.uniform(0.1, 3.0), 4)
        return f"{output_name} = torch.nn.functional.elu({input_name}, alpha={alpha!r})"


class SiLUOperator(Operator):
    # [CLASS DOC] Elementwise silu operator. Stateless.
    """Operator for torch.nn.functional.silu (Sigmoid Linear Unit, also known as Swish)."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the name torch.nn.functional.silu.
        super().__init__("torch.nn.functional.silu")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name torch.nn.functional.silu.
        """Return the torch operation name."""
        return "torch.nn.functional.silu"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs with float dtype (float16, bfloat16,
        # float32, float64).
        """SiLU can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec with the same size, stride and dtype as the
        # output. Raises ValueError if output_spec is not a TensorSpec.
        """Generate input specs for SiLU operation.

        SiLU is element-wise, so input shape matches output shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("SiLUOperator can only produce TensorSpec outputs")

        input_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line applying silu to the single input. Raises ValueError if
        # the number of input names is wrong.
        """Generate code for SiLU operation."""
        if len(input_names) != 1:
            raise ValueError("SiLU requires exactly 1 input")

        input_name = input_names[0]
        return f"{output_name} = torch.nn.functional.silu({input_name})"


class ScaledDotProductAttentionOperator(Operator):
    # [CLASS DOC] Operator for scaled dot product attention. Stateless.
    """Operator for torch.nn.functional.scaled_dot_product_attention."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the
        # name torch.nn.functional.scaled_dot_product_attention.
        super().__init__("torch.nn.functional.scaled_dot_product_attention")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name
        # torch.nn.functional.scaled_dot_product_attention.
        """Return the torch operation name."""
        return "torch.nn.functional.scaled_dot_product_attention"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for float tensor specs with at least three dimensions.
        """Scaled dot product attention can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # SDPA needs at least 3 dimensions (batch, seq_len, embed_dim)
        if len(output_spec.size) < 3:
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns three identical specs (query, key, value), each equal to the
        # output spec. Raises ValueError for non-tensor specs or fewer than three dimensions.
        """Generate input specs for scaled_dot_product_attention.

        SDPA requires:
        - query: (batch, seq_len, embed_dim) or (batch, num_heads, seq_len, head_dim)
        - key: (batch, seq_len, embed_dim) or (batch, num_heads, seq_len_kv, head_dim)
        - value: (batch, seq_len, embed_dim) or (batch, num_heads, seq_len_kv, head_dim)
        Output shape matches query shape.
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                "ScaledDotProductAttentionOperator can only produce TensorSpec outputs"
            )

        if len(output_spec.size) < 3:
            raise ValueError("SDPA output must have at least 3 dimensions")

        # Query has the same shape as output
        query_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        # Key and value: match query shape for simplicity
        # In practice, seq_len for key/value can differ, but we'll keep it simple
        key_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )
        value_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        return [query_spec, key_spec, value_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line calling scaled_dot_product_attention with query, key and
        # value cast to the output dtype. Raises ValueError unless there are exactly three
        # input names.
        """Generate code for scaled_dot_product_attention operation."""
        if len(input_names) != 3:
            raise ValueError("SDPA requires exactly 3 inputs: query, key, value")

        # Ensure dtype compatibility by converting all inputs to the expected output dtype
        target_dtype = str(output_spec.dtype)
        query_name, key_name, value_name = input_names
        return f"{output_name} = torch.nn.functional.scaled_dot_product_attention({query_name}.to({target_dtype}), {key_name}.to({target_dtype}), {value_name}.to({target_dtype}))"


class MultiHeadAttentionForwardOperator(Operator):
    # [CLASS DOC] Operator for multi-head attention forward with a combined input projection and
    # no biases. Stateless.
    """Operator for torch.nn.functional.multi_head_attention_forward."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the
        # name torch.nn.functional.multi_head_attention_forward.
        super().__init__("torch.nn.functional.multi_head_attention_forward")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the torch function name
        # torch.nn.functional.multi_head_attention_forward.
        """Return the torch operation name."""
        return "torch.nn.functional.multi_head_attention_forward"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for float tensor specs with at least three dimensions and
        # no zero-sized dimension.
        """Multi-head attention forward can produce tensor outputs with floating point dtypes."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # MHA needs at least 3 dimensions (seq_len, batch, embed_dim)
        if len(output_spec.size) < 3:
            return False
        # MHA cannot handle 0-sized dimensions (seq_len, batch, or embed_dim must be > 0)
        if any(dim == 0 for dim in output_spec.size):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Treats the first three output dims as sequence, batch and embedding
        # sizes and returns five specs: query, key and value equal to the output spec, a
        # combined input projection weight (3 times embedding by embedding) and an output
        # projection weight (embedding by embedding), all in the output dtype. Raises ValueError
        # for non-tensor specs or fewer than three dimensions.
        # KNOWN ISSUE: the docstring describes optional biases but none are generated. For
        # outputs with more than three dims the embedding size here is the third dim while
        # codegen uses the last dim.
        """Generate input specs for multi_head_attention_forward.

        MHA requires:
        - query, key, value: (seq_len, batch, embed_dim)
        - in_proj_weight: (3*embed_dim, embed_dim) for combined QKV projection
        - in_proj_bias: (3*embed_dim,) optional
        - out_proj_weight: (embed_dim, embed_dim)
        - out_proj_bias: (embed_dim,) optional

        For simplicity, we'll use the combined in_proj_weight path.

        IMPORTANT: The order of optional parameters matters for codegen!
        We must ensure that when we have 6 inputs, they are in the order:
        query, key, value, in_proj_weight, in_proj_bias, out_proj_weight
        NOT: query, key, value, in_proj_weight, out_proj_weight, out_proj_bias
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                "MultiHeadAttentionForwardOperator can only produce TensorSpec outputs"
            )

        if len(output_spec.size) < 3:
            raise ValueError("MHA output must have at least 3 dimensions")

        # Output shape: (seq_len, batch, embed_dim)
        seq_len, batch, embed_dim = output_spec.size[:3]

        # Query, key, value have the same shape as output
        query_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )
        key_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )
        value_spec = TensorSpec(
            size=output_spec.size, stride=output_spec.stride, dtype=output_spec.dtype
        )

        # in_proj_weight: (3*embed_dim, embed_dim)
        in_proj_weight_spec = TensorSpec(
            size=(3 * embed_dim, embed_dim),
            stride=(embed_dim, 1),
            dtype=output_spec.dtype,
        )

        # out_proj_weight: (embed_dim, embed_dim)
        out_proj_weight_spec = TensorSpec(
            size=(embed_dim, embed_dim),
            stride=(embed_dim, 1),
            dtype=output_spec.dtype,
        )

        # For simplicity and correctness, always generate all required tensors
        # This avoids ambiguity in the codegen about which optional parameters are present
        # We'll use a simplified signature: query, key, value, in_proj_weight, out_proj_weight only
        specs = [
            query_spec,
            key_spec,
            value_spec,
            in_proj_weight_spec,
            out_proj_weight_spec,
        ]

        from typing import cast

        return cast(list[Spec], specs)

    def _calculate_stride(self, size):
        # [FUNCTION DOC] HELPER: Returns contiguous strides for a size; unused by this class.
        """Calculate stride for a given size."""
        if not size:
            return ()
        stride = []
        current_stride = 1
        for dim_size in reversed(size):
            stride.append(current_stride)
            current_stride *= dim_size
        return tuple(reversed(stride))

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a call to multi_head_attention_forward (attention weights not
        # computed, eval mode, no dropout, no biases) taking the output's last dimension as
        # embedding size and the largest of 8, 4, 2, 1 dividing it as head count; inputs are
        # cast to the output dtype and the first result is assigned to output_name. Raises
        # ValueError unless there are exactly five input names.
        """Generate code for multi_head_attention_forward operation."""
        if len(input_names) != 5:
            raise ValueError(
                "MHA requires exactly 5 inputs: query, key, value, in_proj_weight, out_proj_weight"
            )

        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                "MultiHeadAttentionForwardOperator can only produce TensorSpec outputs"
            )

        target_dtype = str(output_spec.dtype)
        embed_dim = output_spec.size[-1]

        # Determine number of heads (must divide embed_dim evenly)
        # Common choices: 8, 4, 2, 1
        possible_heads = [h for h in [8, 4, 2, 1] if embed_dim % h == 0]
        num_heads = possible_heads[0] if possible_heads else 1

        query_name = input_names[0]
        key_name = input_names[1]
        value_name = input_names[2]
        in_proj_weight_name = input_names[3]
        out_proj_weight_name = input_names[4]

        # Build the function call without optional biases
        code = f"""{output_name}, _ = torch.nn.functional.multi_head_attention_forward(
    {query_name}.to({target_dtype}),
    {key_name}.to({target_dtype}),
    {value_name}.to({target_dtype}),
    {embed_dim},
    {num_heads},
    {in_proj_weight_name}.to({target_dtype}),
    None,  # in_proj_bias
    None,  # bias_k
    None,  # bias_v
    False,  # add_zero_attn
    0.0,  # dropout_p (no dropout for testing)
    {out_proj_weight_name}.to({target_dtype}),
    None,  # out_proj_bias
    training=False,  # Use eval mode for deterministic behavior
    need_weights=False,  # Don't compute attention weights for performance
)"""

        return code
