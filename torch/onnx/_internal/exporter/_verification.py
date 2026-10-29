# [MODULE DOC]
# Numerical verification of an exported ONNX program against its source ExportedProgram.
# Public API: VerificationInfo (per-value difference statistics) and verify_onnx_program
# (compare final outputs, or every intermediate node value). Private helpers compare
# tensors, map graph value names, and an FX interpreter that checks each node as it runs.
# Import-time side effect: creates a module logger.
from __future__ import annotations


__all__ = [
    "VerificationInfo",
    "verify_onnx_program",
]

import dataclasses
import logging
import math
from typing import Any, TYPE_CHECKING

import torch
from torch.utils import _pytree


if TYPE_CHECKING:
    from torch.onnx._internal._lazy_import import onnx_ir as ir
    from torch.onnx._internal.exporter import _onnx_program


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VerificationInfo:
    # [CLASS DOC]
    # Dataclass record of how one value differs between expected (torch) and actual (ONNX)
    # results: name, maximum absolute and relative difference, histograms of both
    # difference tensors (counts and bin edges) and the two dtypes. Built with from_tensors
    # and exported to plain data with asdict.
    """Verification information for a value in the ONNX program.

    This class contains the maximum absolute difference, maximum relative difference,
    and histograms of absolute and relative differences between the expected and actual
    values. It also includes the expected and actual data types.

    The histograms are represented as tuples of tensors, where the first tensor is the
    histogram counts and the second tensor is the bin edges.

    Attributes:
        name: The name of the value (output or intermediate).
        max_abs_diff: The maximum absolute difference between the expected and actual values.
        max_rel_diff: The maximum relative difference between the expected and actual values.
        abs_diff_hist: A tuple of tensors representing the histogram of absolute differences.
            The first tensor is the histogram counts and the second tensor is the bin edges.
        rel_diff_hist: A tuple of tensors representing the histogram of relative differences.
            The first tensor is the histogram counts and the second tensor is the bin edges.
        expected_dtype: The data type of the expected value.
        actual_dtype: The data type of the actual value.
    """

    name: str
    max_abs_diff: float
    max_rel_diff: float
    abs_diff_hist: tuple[torch.Tensor, torch.Tensor]
    rel_diff_hist: tuple[torch.Tensor, torch.Tensor]
    expected_dtype: torch.dtype
    actual_dtype: torch.dtype
    # NOTE: We don't need to include shape because the expected shape is already known
    # and checked by the runtime

    @classmethod
    def from_tensors(
        cls,
        name: str,
        expected: torch.Tensor | float | int | bool,
        actual: torch.Tensor | float | int | bool,
    ) -> VerificationInfo:
        # [FUNCTION DOC]
        # Class constructor from two values. Python scalars are converted to tensors. Computes
        # max absolute and relative differences and elementwise difference tensors via the
        # comparison helper, then histograms both with fixed bin edges from zero up to one
        # million. Returns a VerificationInfo with the dtypes of the converted inputs.
        """Create a VerificationInfo object from two tensors.

        Args:
            name: The name of the value.
            expected: The expected tensor.
            actual: The actual tensor.

        Returns:
            VerificationInfo: The VerificationInfo object.
        """
        if not isinstance(expected, torch.Tensor):
            expected = torch.tensor(expected)
        if not isinstance(actual, torch.Tensor):
            actual = torch.tensor(actual)

        max_abs_diff, max_rel_diff, abs_diff, rel_diff = _compare_tensors(
            expected, actual
        )
        bins = torch.tensor(
            [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 1000000],
            dtype=torch.float,
        )
        abs_diff_hist = torch.histogram(abs_diff.float(), bins=bins)
        rel_diff_hist = torch.histogram(rel_diff.float(), bins=bins)
        return cls(
            name=name,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
            abs_diff_hist=abs_diff_hist,
            rel_diff_hist=rel_diff_hist,
            expected_dtype=expected.dtype,
            actual_dtype=actual.dtype,
        )

    def asdict(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns a plain mapping of the fields: histograms become lists of counts and bin
        # edges (as nested lists) and dtypes become strings. No side effects.
        """Convert the VerificationInfo object to a dictionary.

        Returns:
            A dictionary representation of the VerificationInfo object.
        """
        return {
            "name": self.name,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "abs_diff_hist": [
                self.abs_diff_hist[0].tolist(),
                self.abs_diff_hist[1].tolist(),
            ],
            "rel_diff_hist": [
                self.rel_diff_hist[0].tolist(),
                self.rel_diff_hist[1].tolist(),
            ],
            "expected_dtype": str(self.expected_dtype),
            "actual_dtype": str(self.actual_dtype),
        }


def _compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Compares two tensors on the CPU (detached). If either is empty,
    # returns infinity for both maxima and infinite scalar difference tensors. Booleans are
    # cast to float32 and a complex expected tensor is viewed as real. Returns (max absolute
    # difference, max relative difference, absolute difference tensor, relative difference
    # tensor), where relative difference divides by the absolute expected value plus 1e-7.
    # KNOWN ISSUE: shapes are assumed broadcast-compatible (a mismatch raises from torch),
    # a bool actual with non-bool expected is not converted, and NaNs are not special-cased.
    # Move tensors to the same device
    expected = expected.detach().cpu()
    actual = actual.detach().cpu()
    if expected.numel() == 0 or actual.numel() == 0:
        return math.inf, math.inf, torch.tensor(math.inf), torch.tensor(math.inf)
    if expected.dtype == torch.bool:
        expected = expected.to(torch.float32)
        actual = actual.to(torch.float32)
    if torch.is_complex(expected):
        expected = torch.view_as_real(expected)
    abs_diff = torch.abs(expected - actual)
    eps = 1e-7
    normalizer = torch.abs(expected) + eps
    rel_diff = abs_diff / normalizer

    max_absolute_difference = abs_diff.max().item()
    max_relative_difference = rel_diff.max().item()

    return max_absolute_difference, max_relative_difference, abs_diff, rel_diff


def verify_onnx_program(
    onnx_program: _onnx_program.ONNXProgram,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    compare_intermediates: bool = False,
) -> list[VerificationInfo]:
    # [FUNCTION DOC]
    # Checks an ONNX program against its ExportedProgram. Uses default example inputs when
    # neither args nor kwargs is given. Raises ValueError if the program has no exported
    # program or no example inputs are available. Without intermediates it runs the torch
    # module and the ONNX program, pairs outputs with graph output names and returns one
    # VerificationInfo per output. With compare_intermediates it runs the verification
    # interpreter and returns infos for every matched node value (outputs included).
    # KNOWN ISSUE: outputs are paired by position with silent truncation to the shortest,
    # and None or complex torch outputs are not adapted as the testing helper does.
    """Verify the ONNX model by comparing the values with the expected values from ExportedProgram.

    Args:
        onnx_program: The ONNX program to verify.
        args: The input arguments for the model.
        kwargs: The keyword arguments for the model.
        compare_intermediates: Whether to verify intermediate values. This is going
            to take longer time, so it is disabled by default.

    Returns:
        VerificationInfo objects containing the verification information for each value.
    """
    exported_program = onnx_program.exported_program
    if exported_program is None:
        raise ValueError(
            "The ONNX program does not contain an exported_program. "
            "Please provide an exported_program to verify the ONNX program."
        )
    if args is None and kwargs is None:
        # User did not provide example inputs, use the default example inputs
        if exported_program.example_inputs is None:
            raise ValueError(
                "No example inputs provided and the exported_program does not contain example inputs. "
                "Please provide arguments to verify the ONNX program."
            )
        args, kwargs = exported_program.example_inputs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    # Flatten args for ONNX program and the VerificationInterpreter
    flat_args, _ = exported_program._get_flat_args_with_check(args, kwargs)

    if not compare_intermediates:
        # Compare the output values
        torch_outputs, _ = _pytree.tree_flatten(
            exported_program.module()(*args, **kwargs)
        )
        onnx_outputs = onnx_program(*flat_args)
        results = []
        for torch_output, onnx_output, output_val in zip(
            torch_outputs, onnx_outputs, onnx_program.model.graph.outputs
        ):
            results.append(
                VerificationInfo.from_tensors(
                    name=str(output_val.name),
                    expected=torch_output,
                    actual=onnx_output,
                )
            )
        return results

    # Use the _VerificationInterpreter to get the intermediate values
    # By design the output values are included too
    interpreter = _VerificationInterpreter(onnx_program)
    interpreter.run(*flat_args)

    return interpreter.verification_infos


def _create_value_mapping(graph: ir.Graph) -> dict[str, ir.Value]:
    # [FUNCTION DOC] HELPER: Builds a mapping from value name to value for the graph
    # initializers, named inputs and named node outputs (unnamed values are skipped).
    # Subgraph values are not included. Returns the mapping.
    """Return a dictionary mapping names to values in the graph.

    The mapping does not include values from subgraphs.

    Args:
        graph: The graph to extract the mapping from.

    Returns:
        A dictionary mapping names to values.
    """
    values: dict[str, ir.Value] = {}
    values.update(graph.initializers)
    # The names of the values can be None or "", which we need to exclude
    for input in graph.inputs:
        if not input.name:
            continue
        values[input.name] = input
    for node in graph:
        for value in node.outputs:
            if not value.name:
                continue
            values[value.name] = value
    return values


class _VerificationInterpreter(torch.fx.Interpreter):
    # [CLASS DOC] HELPER: FX interpreter over the exported module that, after executing each
    # function-call node, computes the same-named value from the ONNX program and records a
    # VerificationInfo. State: the ONNX program, a name-to-value map of its graph, the
    # current flat args, and the verification_infos list (reset on every run).
    """Interpreter for verifying converted ONNX model accuracy by comparing intermediate values.

    To compare models, first initialize the interpreter with an ONNX program.
    Then, call the :meth:`run` method with the input arguments to execute the model.
    The :meth:`run` method will execute the model and populate the
    :attr:`verification_infos` attribute with the verification information for each value.

    ::
        onnx_program = torch.onnx.export(model, args, dynamo=True)
        interpreter = _VerificationInterpreter(onnx_program)
        interpreter.run(*args)
        verification_infos = interpreter.verification_infos
        for info in verification_infos:
            print("value name:", info.name, info)

    The verification information includes the maximum absolute difference, maximum relative
    difference, and histograms of absolute and relative differences between the expected
    and actual values. See :class:`VerificationInfo` for more details.

    Attributes:
        verification_infos: A list of verification information for each value.
            It is populated when the `run` method is called.
    """

    def __init__(self, onnx_program: torch.onnx.ONNXProgram) -> None:
        # [FUNCTION DOC] HELPER: Stores the program, builds the value-name mapping of its
        # graph and initializes empty args and results. Interprets the exported module.
        # Raises ValueError if the program has no exported program.
        """Initialize the _VerificationInterpreter with an ONNX program.

        Args:
            onnx_program: The ONNX program to verify.
        """
        if onnx_program.exported_program is None:
            raise ValueError(
                "The ONNX program does not contain an exported_program. "
                "Please provide an exported_program to verify the ONNX program."
            )
        super().__init__(onnx_program.exported_program.module())
        self._onnx_program = onnx_program
        self._onnx_values = _create_value_mapping(onnx_program.model.graph)
        self._args: tuple[Any, ...] = ()
        self.verification_infos: list[VerificationInfo] = []

    def run(
        self,
        *args: Any,
        initial_env: dict[torch.fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        # [FUNCTION DOC] HELPER: Clears previous results, remembers the input args for ONNX
        # evaluation, then runs the base interpreter. Returns the module result; fills
        # verification_infos as a side effect.
        """Run the interpreter with the given input arguments.

        This method executes the model and populates the :attr:`verification_infos` attribute
        with the verification information for each value.

        Args:
            args: The input arguments for the model.
            initial_env: The initial environment for the interpreter.
            enable_io_processing: Whether to enable IO processing.

        Returns:
            Any: The result of executing the model.
        """
        self.verification_infos = []
        self._args = args
        return super().run(
            *args,
            initial_env=initial_env,
            enable_io_processing=enable_io_processing,
        )

    def run_node(self, n: torch.fx.Node) -> Any:
        # [FUNCTION DOC] HELPER: Executes the node normally. For function-call nodes whose
        # name exists in the ONNX graph, computes that value from the ONNX program on the
        # saved args (failures are logged as warnings and the node is skipped), appends a
        # VerificationInfo and logs it (warning if max abs diff exceeds 0.01 or max rel diff
        # exceeds 0.1, else info). Always returns the original node result.
        result = super().run_node(n)
        if n.op != "call_function":
            return result
        node_name = n.name
        if node_name not in self._onnx_values:
            return result
        try:
            (onnx_result,) = self._onnx_program.compute_values([node_name], self._args)
        except Exception:
            logger.warning(
                "Failed to compute value for node %s", node_name, exc_info=True
            )
            return result
        info = VerificationInfo.from_tensors(
            name=node_name,
            expected=result,
            actual=onnx_result,
        )
        self.verification_infos.append(info)
        if info.max_abs_diff > 0.01 or info.max_rel_diff > 0.1:
            logger.warning(
                "Verification info for node %s: max_abs_diff: %s, max_rel_diff: %s",
                node_name,
                info.max_abs_diff,
                info.max_rel_diff,
            )
        else:
            logger.info(
                "Verification info for node %s: max_abs_diff: %s, max_rel_diff: %s",
                node_name,
                info.max_abs_diff,
                info.max_rel_diff,
            )
        return result
