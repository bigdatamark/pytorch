# [MODULE DOC] Defines KernelInputs, an abstract wrapper over the input nodes of an
# inductor kernel (shapes, strides, dtypes, device, scalars), and MMKernelInputs,
# its matmul-specialized subclass exposing M/N/K, batch size and output layout.
# No import-time side effects beyond imports.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import torch
import torch._inductor.config
from torch._inductor import ir
from torch._inductor.virtualized import V

from .ir import FixedLayout, FlexibleLayout, Layout


if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy


class KernelInputs(ABC):
    # [CLASS DOC] Abstract holder of a non-empty list of input IR nodes plus optional
    # named scalars and an optional output dtype. Offers read-only queries (shapes,
    # strides, dtypes, device) and caches the device name lazily. Subclasses must
    # implement out_dtype and output_layout.
    """
    Class to store and provide access to input nodes for kernels.
    This class takes in a tuple of input nodes and provides methods to access
    information about these nodes, such as their device type and device.
    """

    def __init__(
        self,
        input_nodes: list[Any],
        scalars: dict[str, float | int] | None = None,
        out_dtype: torch.dtype | None = None,
    ):
        # [FUNCTION DOC] Stores the input nodes, scalars (empty mapping if None) and
        # optional output dtype; device name starts unset.
        # Raises AssertionError if input_nodes is empty.
        """
        Initialize with a tuple of input nodes.

        Args:
            input_nodes: A tuple of input nodes to store
            out_dtype: Optional output dtype to store
        """
        self._input_nodes = input_nodes
        self._device_name: str | None = None
        self._scalars = scalars if scalars is not None else {}
        self._out_dtype = out_dtype
        if len(input_nodes) <= 0:
            raise AssertionError("Expected at least one input node")

    def nodes(self, reorder: Sequence[int] | None = None) -> list[Any]:
        # [FUNCTION DOC] Returns the stored input node list (same object, not a copy)
        # or, if reorder is given, a new list picking nodes by those indices.
        # Raises AssertionError if reorder length differs from the node count.
        """
        Return the stored input nodes, optionally reordered.

        Args:
            reorder: Optional sequence of indices to reorder the nodes.
                    For example, (2, 0, 1) would return nodes in that order.

        Returns:
            The tuple of input nodes, optionally reordered
        """
        if reorder is None:
            return self._input_nodes
        if len(self._input_nodes) != len(reorder):
            raise AssertionError(
                f"Reorder length mismatch: {len(self._input_nodes)} vs {len(reorder)}"
            )
        return [self._input_nodes[i] for i in reorder]

    @property
    def count(self) -> int:
        # [FUNCTION DOC] Property: number of stored input nodes.
        """
        Get the number of input nodes.

        Returns:
            The number of input nodes
        """
        return len(self._input_nodes)

    @property
    def device_type(self) -> str | None:
        # [FUNCTION DOC] Property: device type string (e.g. cuda, cpu) of the first
        # input node, via ir.get_device_type; may be None.
        """
        Get the device type of the first node.

        Returns:
            The device type (e.g., 'cuda', 'cpu')
        """

        return ir.get_device_type(self._input_nodes[0])

    def device(self) -> torch.device:
        # [FUNCTION DOC] Returns the torch device of the first input node.
        """
        Get the device of the first node.

        Returns:
            The device of the first node
        """
        return self._input_nodes[0].get_device()

    def device_name(self) -> str | None:
        # [FUNCTION DOC] Returns the cached device name; on first call, if the device
        # type is cuda, queries the device properties and caches the gcnArchName
        # attribute (an AMD/ROCm architecture string). Stays None for other devices.
        # KNOWN ISSUE: docstring promises a (gpu_name, vendor, model) tuple, but a
        # single string or None is returned; gcnArchName may be absent on non-ROCm CUDA.
        """
        Get the device name information.

        Returns:
            A tuple of (gpu_name, vendor, model)
        """
        if self._device_name is None:
            device = self.device()
            if self.device_type == "cuda":
                device_properties = torch.cuda.get_device_properties(device)
                self._device_name = device_properties.gcnArchName
        return self._device_name

    def shapes_symbolic(self) -> tuple[tuple[Any, ...], ...]:
        # [FUNCTION DOC] Returns one (possibly symbolic) size tuple per input node.
        """
        Get the symbolic shapes of all input nodes.

        Returns:
            A tuple of shape tuples for each input node
        """
        return tuple(node.get_size() for node in self._input_nodes)

    def shapes_hinted(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] Returns one tuple of concrete integer size hints per input
        # node, obtained from the current graph's sizevars optimization hints.
        # Requires an active graph (V.graph).
        """
        Get the size hints for shapes of all input nodes.

        Returns:
            A tuple of shape tuples with integer hints for each input node
        """
        return tuple(
            V.graph.sizevars.optimization_hints(node.get_size())
            for node in self._input_nodes
        )

    def strides_symbolic(self) -> tuple[tuple[sympy.Integer, ...], ...]:
        # [FUNCTION DOC] Returns one (possibly symbolic) stride tuple per input node.
        """
        Get the symbolic strides of all input nodes.

        Returns:
            A tuple of stride tuples for each input node
        """
        return tuple(node.get_stride() for node in self._input_nodes)

    def strides_hinted(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] Returns one tuple of concrete integer stride hints per input
        # node via the current graph's sizevars. Requires an active graph.
        """
        Get the size hints for strides of all input nodes.

        Returns:
            A tuple of stride tuples with integer hints for each input node
        """
        return tuple(
            V.graph.sizevars.optimization_hints(node.get_stride())
            for node in self._input_nodes
        )

    def dtypes(self) -> tuple[torch.dtype, ...]:
        # [FUNCTION DOC] Returns the dtype of every input node as a tuple.
        """
        Get the dtypes of all input nodes.

        Returns:
            A tuple of dtypes for each input node
        """
        return tuple(node.get_dtype() for node in self._input_nodes)

    def dtype(self, idx: int = 0) -> torch.dtype:
        # [FUNCTION DOC] Returns the dtype of the input node at idx (default first).
        # Raises IndexError if idx is out of range.
        """
        Get the dtype of a specific input node.

        Args:
            idx: Index of the node to get the dtype from (default: 0)

        Returns:
            The dtype of the specified input node
        """
        return self._input_nodes[idx].get_dtype()

    @abstractmethod
    def out_dtype(self) -> torch.dtype:
        # [FUNCTION DOC] Abstract: subclasses return the output dtype, either the
        # explicitly provided one or one inferred from the input nodes.
        """
        Get the output dtype, whether passed in or inferred from the nodes

        Returns:
            The output dtype
        """

    def get_scalar(self, name: str) -> float | int:
        # [FUNCTION DOC] Returns the named scalar from the stored mapping.
        # Raises AssertionError if the name is missing.
        """
        Get the scalar value for a given name.

        Args:
            name: Name of the scalar to get

        Returns:
            The scalar value
        """
        if name not in self._scalars:
            raise AssertionError(f"Scalar {name} not found, but required")
        return self._scalars[name]

    @abstractmethod
    def output_layout(self, flexible: bool = True) -> Layout:
        # [FUNCTION DOC] Abstract: subclasses build the output Layout, flexible
        # (FlexibleLayout) when flexible is true, else fixed (FixedLayout).
        # KNOWN ISSUE: docstring lists an out_dtype argument that does not exist.
        """
        Abstract method to handle output layout generation.

        Args:
            out_dtype: Optional output dtype. If not provided, infer from inputs
            flexible: If True, return FlexibleLayout, otherwise FixedLayout
        """


class MMKernelInputs(KernelInputs):
    # [CLASS DOC] KernelInputs for matmul-like ops. Tracks which input positions are
    # mat1 and mat2 (default last two; negative indices allowed) so extra inputs such
    # as bias or scales can coexist. Provides M/N/K (symbolic and hinted), batch
    # size, mat accessors, output dtype and output layout.
    """
    Specialized KernelInputs for matrix multiplication operations.
    Provides additional methods to access M, N, K dimensions.
    """

    def __init__(
        self,
        input_nodes: list[Any],
        scalars: dict[str, float | int] | None = None,
        out_dtype: torch.dtype | None = None,
        mat1_idx: int = -2,
        mat2_idx: int = -1,
    ):
        # [FUNCTION DOC] Initializes the base class, then validates and stores the
        # mat1/mat2 positions (negative values count from the end).
        # Raises AssertionError if fewer than 2 nodes or either index is out of range.
        # Note the stored indices are the originals (possibly negative), not resolved.
        """
        Initialize with a tuple of input nodes.

        By default, we assume the last 2 input nodes are mat1 and mat2, but
        the caller can adjust when necessary
        """
        super().__init__(input_nodes, scalars, out_dtype)
        # for mm, we need at least 2 nodes, and we need to know which nodes
        # are the main matrices e.g. addmm is (bias, mat1, mat2) whereas others
        # might be (mat1, mat2, scale), etc.
        if len(self._input_nodes) < 2:
            raise AssertionError("Expected at least 2 input nodes")

        # Adjust assertions to handle negative indices
        m1_idx, m2_idx = mat1_idx, mat2_idx
        if mat1_idx < 0:
            m1_idx += len(input_nodes)
        if mat2_idx < 0:
            m2_idx += len(input_nodes)

        if not (0 <= m1_idx < len(input_nodes)):
            raise AssertionError(f"Invalid mat1_idx: {mat1_idx}")
        if not (0 <= m2_idx < len(input_nodes)):
            raise AssertionError(f"Invalid mat2_idx: {mat2_idx}")

        self._mat1_idx = mat1_idx
        self._mat2_idx = mat2_idx

    def mnk_symbolic(
        self,
    ) -> tuple[sympy.Integer, sympy.Integer, sympy.Integer]:
        # [FUNCTION DOC] Returns symbolic (M, N, K): M and K from the last two dims of
        # mat1, N from the last dim of mat2; leading batch dims are ignored. Asserts
        # via sizevars that mat1's K equals mat2's second-to-last dim (may add a guard).
        """
        Get the symbolic M, N, K dimensions for matrix multiplication.
        Handles both 2D (MM) and 3D (BMM) tensors.

        M is extracted from the second-to-last dimension of the first operand (mat1).
        N is extracted from the last dimension of the second operand (mat2).
        K is extracted from the last dimension of the first operand (mat1).

        Returns:
            A tuple of (M, N, K) dimensions
        """
        mat1 = self.nodes()[self._mat1_idx]
        mat2 = self.nodes()[self._mat2_idx]

        m = mat1.get_size()[-2]  # M from second-to-last dimension of mat1
        k = mat1.get_size()[-1]  # K from last dimension of mat1
        n = mat2.get_size()[-1]  # N from last dimension of mat2

        # Ensure K dimensions match between operands
        k0 = mat2.get_size()[-2]  # K from second-to-last dimension of mat2
        V.graph.sizevars.check_equals(k, k0)
        return (m, n, k)

    def out_dtype(self) -> torch.dtype:
        # [FUNCTION DOC] Returns the explicit output dtype if given, else the dtype of mat1.
        """
        Get the output dtype, whether passed in or inferred from the nodes

        Returns:
            The output dtype
        """
        if self._out_dtype is not None:
            return self._out_dtype
        return self.mat1mat2()[0].get_dtype()

    def output_layout(self, flexible: bool = True) -> Layout:
        # [FUNCTION DOC] Builds the matmul output layout on the first node's device
        # with out_dtype(); size is the broadcast-checked batch dims of mat1/mat2
        # (equality enforced/simplified through sizevars) followed by M and N.
        # Returns FlexibleLayout if flexible else FixedLayout.
        # KNOWN ISSUE: docstring mentions a nonexistent out_dtype argument; the k1/k2
        # dims are unpacked but never compared here.
        """
        Handle output layout generation for matrix multiplication.

        Args:
            out_dtype: Optional output dtype. If not provided, infer from inputs
            flexible: If True, return FlexibleLayout, otherwise FixedLayout
        """
        mat1, mat2 = self.mat1mat2()
        out_dtype = self.out_dtype()
        # NOTE: taken from mm_common.mm_args
        *b1, m, k1 = mat1.get_size()
        *b2, k2, n = mat2.get_size()
        b = [V.graph.sizevars.check_equals_and_simplify(a, b) for a, b in zip(b1, b2)]
        size = [*b, m, n]
        if flexible:
            return FlexibleLayout(self.device(), out_dtype, size)
        else:
            return FixedLayout(self.device(), out_dtype, size)

    def mat1mat2(self) -> tuple[Any, Any]:
        # [FUNCTION DOC] Returns the (mat1, mat2) input nodes at the configured positions.
        """
        Get the mat1 and mat2 nodes.

        Returns:
            A tuple of (mat1, mat2) nodes
        """
        nodes = self.nodes()
        return nodes[self._mat1_idx], nodes[self._mat2_idx]

    def mnk_hinted(self) -> tuple[int, int, int]:
        # [FUNCTION DOC] Returns integer-hinted (M, N, K) using shapes_hinted, same
        # dimension convention as mnk_symbolic. Raises AssertionError if the K hints differ.
        """
        Get the hinted M, N, K dimensions for matrix multiplication.
        Handles both 2D (MM) and 3D (BMM) tensors.

        Uses shapes_hinted from the base class to get integer hints for dimensions.

        Returns:
            A tuple of (M, N, K) dimensions as integers
        """
        hinted_shapes = self.shapes_hinted()
        mat1_shape = hinted_shapes[self._mat1_idx]
        mat2_shape = hinted_shapes[self._mat2_idx]

        m = mat1_shape[-2]  # M from second-to-last dimension of mat1
        k = mat1_shape[-1]  # K from last dimension of mat1
        n = mat2_shape[-1]  # N from last dimension of mat2

        # Ensure K dimensions match between operands
        k_check = mat2_shape[-2]  # K from second-to-last dimension of mat2
        if k != k_check:
            raise AssertionError(f"K dimensions don't match: {k} vs {k_check}")

        return (m, n, k)

    def batch_hinted(self) -> int:
        # [FUNCTION DOC] Returns the hinted size of the third-from-last dim of mat1
        # if mat1 has at least 3 dims, else 1. Only one batch dim is considered.
        """
        Get the hinted batch size for batched matrix multiplication.
        Returns 1 for non-batched (2D) operations.

        Returns:
            The batch size as an integer
        """
        hinted_shapes = self.shapes_hinted()
        mat1_shape = hinted_shapes[self._mat1_idx]

        if len(mat1_shape) >= 3:
            return mat1_shape[-3]  # Batch from third-to-last dimension
        return 1  # Non-batched operation
