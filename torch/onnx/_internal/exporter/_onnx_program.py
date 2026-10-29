# mypy: allow-untyped-defs
# mypy: disable-error-code="attr-defined,name-defined"
# [MODULE DOC] Defines ONNXProgram, the public result of torch.onnx.export: an ONNX IR model
# plus the ExportedProgram it came from, callable with torch tensors through ONNX Runtime or the
# reference evaluator, and savable to disk. Also holds helpers converting between torch tensors,
# NumPy arrays and ONNX Runtime values (including bfloat16 and 8-bit float types), flattening
# call arguments, and handling large models. Defines a size threshold of 1.5GB for switching to
# external data, a set of float8 dtypes unsupported by NumPy, and a logger. No heavy imports at
# import time (onnxruntime and others are imported lazily).
from __future__ import annotations


__all__ = ["ONNXProgram"]

import contextlib
import copy
import gc
import logging
import os
import tempfile
import textwrap
import warnings
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING

import torch
from torch.onnx._internal._lazy_import import onnx, onnx_ir as ir, onnxscript_apis
from torch.onnx._internal.exporter import _dynamic_shapes, _ir_passes
from torch.utils import _pytree


# NOTE: DO NOT import module from torch.onnx._internal to this module in the global scope
# because ONNXProgram is exposed to the public API

if TYPE_CHECKING:
    import numpy as np
    import onnxruntime as ort

_LARGE_MODEL_THRESHOLD = 1536 * 1024 * 1024  # 1536MB
_NP_UNSUPPORTED_DTYPES_8BIT = frozenset(
    {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }
)

logger = logging.getLogger(__name__)


def _ort_session_initializer(model: str | bytes) -> ort.InferenceSession:
    # [FUNCTION DOC] HELPER: Default session factory. Creates an ONNX Runtime session from a
    # model (serialized bytes or a file path), with error-level logging, preferring the CUDA
    # provider over the CPU provider when available. Raises ImportError if onnxruntime is
    # missing.
    """Initialize an ONNX Runtime inference session with the specified model."""
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # 3: Error
    possible_providers = (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    available_providers = set(ort.get_available_providers())
    providers = [
        provider for provider in possible_providers if provider in available_providers
    ]
    return ort.InferenceSession(
        model, providers=providers, sess_options=session_options
    )


def _count_initializer_size(graph: ir.Graph) -> int:
    # [FUNCTION DOC] HELPER: Returns the total size in bytes of all initializers that have a
    # constant value.
    """Count the total size of the initializers in bytes."""
    return sum(
        v.const_value.nbytes
        for v in graph.initializers.values()
        if v.const_value is not None
    )


@contextlib.contextmanager
def _set_graph_outputs(
    graph: ir.Graph,
    outputs: list[ir.Value],
):
    # [FUNCTION DOC] HELPER: Context manager that temporarily replaces the outputs of a
    # graph by the given values and restores the original outputs on exit, even on error.
    # Mutates the graph.
    """Temporarily set the outputs of the graph.

    Args:
        graph: The graph to set the outputs for.
        outputs: The outputs to set.
    """
    original_outputs = list(graph.outputs)
    graph.outputs.clear()
    graph.outputs.extend(outputs)
    try:
        yield
    finally:
        graph.outputs.clear()
        graph.outputs.extend(original_outputs)


def _create_value_mapping(graph: ir.Graph) -> dict[str, ir.Value]:
    # [FUNCTION DOC] HELPER: Builds a mapping from name to value for initializers, graph
    # inputs and node outputs of the top-level graph (not subgraphs), skipping unnamed
    # values. Later entries override earlier ones with the same name.
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


def _to_numpy_array(input: torch.Tensor | int | float | str | bool) -> np.ndarray:
    # [FUNCTION DOC] HELPER: Converts a Python scalar or a torch tensor to a NumPy array,
    # using the exporter tensor wrapper for tensors so special dtypes are handled.
    if isinstance(input, (int, float, str, bool)):
        return ir.tensor(input).numpy()

    from torch.onnx._internal.exporter import _core

    return _core.TorchTensor(input).numpy()


def _from_numpy_array(array: np.ndarray) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Converts a NumPy array to a torch tensor. Arrays of bfloat16 or
    # float8 types from the ml_dtypes package are reinterpreted bit-wise into the matching
    # torch dtypes; others use the standard conversion.
    """Convert a NumPy array to a PyTorch tensor."""
    import ml_dtypes  # type: ignore[import-not-found]
    import numpy as np

    if array.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    if array.dtype == ml_dtypes.float8_e4m3fn:
        return torch.from_numpy(array.view(np.uint8)).view(torch.float8_e4m3fn)
    if array.dtype == ml_dtypes.float8_e4m3fnuz:
        return torch.from_numpy(array.view(np.uint8)).view(torch.float8_e4m3fnuz)
    if array.dtype == ml_dtypes.float8_e5m2:
        return torch.from_numpy(array.view(np.uint8)).view(torch.float8_e5m2)
    if array.dtype == ml_dtypes.float8_e5m2fnuz:
        return torch.from_numpy(array.view(np.uint8)).view(torch.float8_e5m2fnuz)
    return torch.from_numpy(array)


def _to_ort_value(input: torch.Tensor | int | float | str | bool) -> ort.OrtValue:
    # [FUNCTION DOC] HELPER: Converts a Python scalar or torch tensor to an ONNX Runtime
    # value. Integers become 64-bit and floats 32-bit arrays. Bfloat16 and float8 tensors
    # are viewed as unsigned integers and passed with an explicit ONNX element type. Raises
    # RuntimeError if that is unsupported by the installed onnxruntime. Other tensors go
    # through NumPy.
    """Convert a PyTorch tensor to an ONNX Runtime OrtValue."""
    import numpy as np
    import onnxruntime as ort

    from torch.onnx._internal.exporter import _core

    if isinstance(input, (int, float, str, bool)):
        # Convert scalar values to OrtValue
        dtype_mapping = {
            int: np.int64,
            float: np.float32,
        }
        # pyrefly: ignore [bad-argument-type, no-matching-overload]
        dtype = dtype_mapping.get(type(input))
        return ort.OrtValue.ortvalue_from_numpy(np.array(input, dtype=dtype))

    if input.dtype == torch.bfloat16 or input.dtype in _NP_UNSUPPORTED_DTYPES_8BIT:
        if hasattr(ort.OrtValue, "ortvalue_from_numpy_with_onnx_type"):
            # This requires ONNX Runtime 1.21 or newer
            if input.dtype == torch.bfloat16:
                uint_type = torch.uint16
            else:
                uint_type = torch.uint8
            onnx_type = _core.torch_dtype_to_onnx_dtype(input.dtype)
            # Make tensor contiguous to ensure view() works
            input = input.contiguous()
            return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                input.view(uint_type).numpy(force=True), onnx_element_type=onnx_type
            )
        raise RuntimeError(
            f"Failed to convert tensor of type '{input.dtype}' to OrtValue. "
            "Please ensure that ONNX Runtime is built with DLPack support or is the latest version"
        )
    # TODO(#151064): Use dlpack when ORT properly supports it
    return ort.OrtValue.ortvalue_from_numpy(input.numpy(force=True))


def _from_ort_value(value: ort.OrtValue) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Converts an ONNX Runtime value to a torch tensor. Bfloat16 and
    # float8 values use DLPack; failure raises RuntimeError. Other types go through NumPy.
    if value.element_type() in (
        ir.DataType.BFLOAT16,
        ir.DataType.FLOAT8E4M3FN,
        ir.DataType.FLOAT8E4M3FNUZ,
        ir.DataType.FLOAT8E5M2,
        ir.DataType.FLOAT8E5M2FNUZ,
    ):
        # This requires ONNX Runtime 1.21 or newer
        try:
            return torch.from_dlpack(value._get_c_value())
        except Exception as e:
            raise RuntimeError(
                "Failed to convert OrtValue to torch.Tensor. "
                "Please ensure that ONNX Runtime is built with DLPack support or is the latest version"
            ) from e
    return torch.from_numpy(value.numpy())


class ONNXProgram:
    # [CLASS DOC] Public result of exporting a model to ONNX; callable with the same inputs
    # as the original model. State: model (ONNX IR model, mutable), exported_program
    # (source, may be none), a lazily created ONNX Runtime session, a temporary directory
    # used for large models, and the capture strategy name. Key methods: __call__,
    # call_reference, compute_values, save, optimize, apply_weights, rename_axes,
    # initialize_inference_session, release, and the model_proto property.
    """A class to represent an ONNX program that is callable with torch tensors.

    Attributes:
        model: The ONNX model as an ONNX IR model object.
        exported_program: The exported program that produced the ONNX model.
    """

    def __init__(
        self, model: ir.Model, exported_program: torch.export.ExportedProgram | None
    ) -> None:
        # [FUNCTION DOC] Stores the model and optional exported program; no session is
        # created yet, no temporary directory and no capture strategy.
        """Initialize the ONNX program with the specified model and exported program.
        Args:
            model: The ONNX model.
            exported_program: The exported program that produced the ONNX model. Optional.
        """
        self.model: ir.Model = model
        self.exported_program = exported_program
        self._inference_session: ort.InferenceSession | None = None
        self._tempdir: tempfile.TemporaryDirectory | None = None
        # Strategy used to capture the exported program
        self._capture_strategy: str | None = None

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a multi-line text showing the model and the exported
        # program, each indented.
        return f"""\
ONNXProgram(
    model=
{textwrap.indent(str(self.model), " " * 8)}
    ,
    exported_program=
{textwrap.indent(str(self.exported_program), " " * 8)}
)
"""

    def __call__(self, *args, **kwargs) -> Sequence[torch.Tensor]:
        # [FUNCTION DOC] Runs the model with ONNX Runtime. Args and kwargs are
        # flattened, none values dropped and complex tensors converted to real form,
        # then matched in order to the graph inputs. Creates the session on first use.
        # Returns: tuple of output tensors. Raises ImportError if onnxruntime is
        # missing.
        # KNOWN ISSUE: inputs are paired with graph inputs by position and extra or
        # missing values are silently ignored by the pairing.
        """Run the ONNX model with the same arguments you would provide to the GraphModule."""
        import onnxruntime as ort

        flatten_args = _process_args(args, kwargs)

        if self._inference_session is None:
            self.initialize_inference_session()

        if self._inference_session is None:
            raise AssertionError("_inference_session must be non-None")

        ort_input = {
            k.name: _to_ort_value(v)
            for k, v in zip(self.model.graph.inputs, flatten_args)
        }
        run_options = ort.RunOptions()
        run_options.log_severity_level = 3  # 3: Error
        logger.debug("Running the inference session with %s arguments.", len(ort_input))
        outputs = self._inference_session.run_with_ort_values(
            None, ort_input, run_options=run_options
        )
        logger.debug("Inference session run completed.")
        return tuple(_from_ort_value(output) for output in outputs)

    def call_reference(self, *args, **kwargs) -> Sequence[torch.Tensor]:
        # [FUNCTION DOC] Runs the model with the ONNX reference evaluator on NumPy
        # arrays, using the same input preparation as the call operator. Returns: tuple
        # of output tensors. Raises AssertionError if the evaluator does not return a
        # sequence.
        """Run the ONNX model using the reference backend."""
        import onnx.reference

        evaluator = onnx.reference.ReferenceEvaluator(self.model_proto)

        flatten_args = _process_args(args, kwargs)
        ref_input = {
            k.name: _to_numpy_array(v)
            for k, v in zip(self.model.graph.inputs, flatten_args)
        }
        outputs = evaluator.run(None, ref_input)  # type: ignore[arg-type]
        if not isinstance(outputs, Sequence):
            raise AssertionError(f"Expected Sequence, got {type(outputs)}")
        return tuple(_from_numpy_array(output) for output in outputs)

    def compute_values(
        self, value_names: Sequence[str], args=(), kwargs=None
    ) -> Sequence[torch.Tensor]:
        # [FUNCTION DOC] Computes intermediate values by name. Temporarily makes the
        # named values the graph outputs, runs the model on args and kwargs, and
        # restores outputs. Releases any existing session before and after, so the next
        # call creates a new session. Args: value_names, args, kwargs. Returns: tuple of
        # tensors in the order of value_names. Raises ValueError if a name is not found
        # in the top-level graph.
        # KNOWN ISSUE: the docstring says it returns a dictionary and omits args and
        # kwargs; the code returns a tuple.
        """Compute the values of the specified names in the ONNX model.

        This method is used to compute the values of the specified names in the ONNX model.
        The values are returned as a dictionary mapping names to tensors.

        Args:
            value_names: The names of the values to compute.

        Returns:
            A dictionary mapping names to tensors.
        """
        if kwargs is None:
            kwargs = {}
        self.release()
        values = _create_value_mapping(self.model.graph)
        for name in value_names:
            if name not in values:
                raise ValueError(
                    f"Value '{name}' not found in the model. "
                    "Please provide a valid value name."
                )
        temporary_outputs = [values[name] for name in value_names]
        with _set_graph_outputs(self.model.graph, temporary_outputs):
            try:
                result = self(*args, **kwargs)
            finally:
                self.release()
        return result

    @property
    def model_proto(self) -> onnx.ModelProto:
        # [FUNCTION DOC] Property. Serializes the current IR model into a new ONNX
        # ModelProto each time.
        """Return the ONNX ``ModelProto`` object."""
        return ir.serde.serialize_model(self.model)

    def optimize(self) -> None:
        # [FUNCTION DOC] Replaces the model by an optimized version (constant folding,
        # redundancy removal) from onnxscript. Does not reset an existing inference
        # session.
        """Optimize the ONNX model.

        This method optimizes the ONNX model by performing constant folding and
        eliminating redundancies in the graph. The optimization is done in-place.
        """
        self.model = onnxscript_apis.optimize(self.model)

    def save(
        self,
        destination: str | os.PathLike,
        *,
        include_initializers: bool = True,
        keep_initializers_as_inputs: bool = False,
        external_data: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Writes the model to destination. Optional flags control whether
        # initializers are included and whether they also stay as graph inputs.
        # Temporarily mutates the graph and restores it afterwards, even on error. Uses
        # external data files when requested or when the initializers exceed 1.5GB;
        # otherwise saves as one file. Args: destination, include_initializers,
        # keep_initializers_as_inputs, external_data. Side effects: file I/O.
        # KNOWN ISSUE: the docstring says it raises TypeError for a non-path destination
        # with external data; the function does not check this itself.
        """Save the ONNX model to the specified destination.

        When ``external_data`` is ``True`` or the model is larger than 2GB,
        the weights are saved as external data in a separate file.

        Initializer (model weights) serialization behaviors:

        * ``include_initializers=True``, ``keep_initializers_as_inputs=False`` (default):
          The initializers are included in the saved model.
        * ``include_initializers=True``, ``keep_initializers_as_inputs=True``:
          The initializers are included in the saved model and kept as model inputs.
          Choose this option if you want the ability to override the model weights
          during inference.
        * ``include_initializers=False``, ``keep_initializers_as_inputs=False``:
          The initializers are not included in the saved model and are not listed
          as model inputs. Choose this option if you want to attach the initializers
          to the ONNX model in a separate, post-processing, step.
        * ``include_initializers=False``, ``keep_initializers_as_inputs=True``:
          The initializers are not included in the saved model but are listed as model
          inputs. Choose this option if you want to supply the initializers during
          inference and want to minimize the size of the saved model.

        Args:
            destination: The path to save the ONNX model to.
            include_initializers: Whether to include the initializers in the saved model.
            keep_initializers_as_inputs: Whether to keep the initializers as inputs in the saved model.
                If `True`, the initializers are added as inputs to the model which means they can be overwritten.
                by providing the initializers as model inputs.
            external_data: Whether to save the weights as external data in a separate file.

        Raises:
            TypeError: If ``external_data`` is ``True`` and ``destination`` is not a file path.
        """
        original_initializers = copy.copy(self.model.graph.initializers)
        original_inputs = copy.copy(self.model.graph.inputs)

        # Adjust the model based on options
        if not include_initializers:
            self.model.graph.initializers.clear()
        if keep_initializers_as_inputs:
            self.model.graph.inputs.extend(original_initializers.values())  # type: ignore[arg-type]

        try:
            # Save the model to disk
            if (
                external_data
                or _count_initializer_size(self.model.graph) > _LARGE_MODEL_THRESHOLD
            ):
                onnxscript_apis.save_model_with_external_data(self.model, destination)
            else:
                ir.save(self.model, destination)
        finally:
            # Revert the changes to the model
            if not include_initializers:
                self.model.graph.initializers.update(original_initializers)
            if keep_initializers_as_inputs:
                self.model.graph.inputs.clear()
                self.model.graph.inputs.extend(original_inputs)

    def apply_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        # [FUNCTION DOC] Replaces initializer values with tensors from state_dict,
        # matched by name. Names not in the model are skipped with an exporter warning.
        # Mutates the model.
        """Apply the weights from the specified state dict to the ONNX model.

        Use this method to replace FakeTensors or other weights.

        Args:
            state_dict: The state dict containing the weights to apply to the ONNX model.
        """
        from torch.onnx._internal.exporter import _core

        for name, tensor in state_dict.items():
            if name in self.model.graph.initializers:
                self.model.graph.initializers[name].const_value = _core.TorchTensor(
                    tensor, name
                )
            else:
                warnings.warn(
                    f"Weight '{name}' not found in the model. Skipped applying.",
                    category=torch.onnx.errors.OnnxExporterWarning,
                    stacklevel=1,
                )

    def initialize_inference_session(
        self,
        initializer: Callable[
            [str | bytes], ort.InferenceSession
        ] = _ort_session_initializer,
    ) -> None:
        # [FUNCTION DOC] Creates the ONNX Runtime session via initializer and stores it.
        # Models with initializers over 1.5GB are saved to a temporary directory with
        # external data and the file path is passed; otherwise the serialized bytes are
        # passed. Replaces any previous session. Side effects: may create a temporary
        # directory, kept until release.
        """Initialize the ONNX Runtime inference session.

        Args:
            initializer: The function to initialize the ONNX Runtime inference
                session with the specified model. By default, it uses the
                :func:`_ort_session_initializer` function.
        """
        # TODO(justinchuby): Allow different inference options
        logger.debug("Initializing the inference session.")
        if (
            byte_size := _count_initializer_size(self.model.graph)
        ) > _LARGE_MODEL_THRESHOLD:
            logger.debug("The model initializers is larger than 1.5GB (%s).", byte_size)
            # Save the model to a temporary file if too large
            self._tempdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            model_path = os.path.join(self._tempdir.name, "model.onnx")
            self.save(model_path, external_data=True)
            model = model_path
        else:
            model = self.model_proto.SerializeToString()  # type: ignore[assignment]

        self._inference_session = initializer(model)
        logger.debug("Inference session initialized.")

    def release(self) -> None:
        # [FUNCTION DOC] Drops the inference session, runs garbage collection and
        # deletes the temporary directory, if any. Safe to call repeatedly.
        """Release the inference session.

        You may call this method to release the resources used by the inference session.
        """
        # Release the inference session first so that the model file can be deleted
        if self._inference_session is not None:
            self._inference_session = None
        gc.collect()
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def rename_axes(self, rename_mapping: dict[str | ir.SymbolicDim, str]) -> None:
        # [FUNCTION DOC] Renames symbolic dimensions in the model using a mapping from
        # names or symbolic dimensions to new names. Mutates the model; errors come from
        # the rename pass.
        """Rename axes in a model according to the specified rename mapping.

        Example::

            batch = onnx_program.model.graph.inputs[0].shape[0]
            seq_len = onnx_program.model.graph.inputs[0].shape[2]
            rename_mapping = {
                batch: "batch",
                seq_len: "seq_len",
            }
            onnx_program.rename_axes(rename_mapping)

        Args:
            rename_mapping: A dictionary mapping old axes to new axis names.
                Keys can be either:

                * String axis names (e.g., "s1", "s2")
                * SymbolicDim objects obtained from the model
                  (e.g., onnx_program.model.graph.inputs[0].shape[0])

                Values must be strings representing the new axis names.
        """
        _ir_passes.rename_axis(self.model, rename_mapping)

    def _rename_dynamic_axes(
        self,
        dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any],
    ) -> None:
        # [FUNCTION DOC] HELPER: Builds a rename mapping from dynamic_shapes and the
        # graph inputs and applies it to the model, so custom axis names from the user
        # are used.
        """Rename dynamic axes in a model according to the specified dynamic_axes names."""
        rename_mapping = _dynamic_shapes.create_rename_mapping(
            self.model.graph.inputs, dynamic_shapes
        )
        _ir_passes.rename_axis(self.model, rename_mapping)


def _process_args(args, kwargs) -> tuple[torch.Tensor, ...]:
    # [FUNCTION DOC] HELPER: Prepares call arguments for the ONNX model: flattens them,
    # removes none values, and converts complex tensors to real form. Returns a tuple of
    # values.
    """Process input arguments for the ONNX model."""
    args = _flatten_inputs(args, kwargs)
    args = _remove_none_from_inputs(args)
    args = _convert_complex_to_real_representation(args)
    return args


def _flatten_inputs(model_args, model_kwargs):
    # [FUNCTION DOC] HELPER: Flattens positional and keyword arguments with a pytree and
    # returns the list of leaves.
    flattened_args, _ = _pytree.tree_flatten((model_args, model_kwargs))
    return flattened_args


def _remove_none_from_inputs(model_args):
    # [FUNCTION DOC] HELPER: Returns a tuple of the arguments without the none entries.
    return tuple(arg for arg in model_args if arg is not None)


def _convert_complex_to_real_representation(model_args):
    # [FUNCTION DOC] HELPER: Returns a tuple where each complex tensor is replaced by its
    # real view with a trailing dimension of size two (conjugates resolved first); others
    # pass unchanged.
    """Convert complex dtype tensors to real representation tensors.

    ONNX does not support complex dtype tensors. Thus, we convert complex dtype tensors
    to real representation tensors (i.e., float dtype tensors with an extra dimension
    representing the real and imaginary parts of the complex number).
    """
    return tuple(
        torch.view_as_real(arg.resolve_conj())
        if isinstance(arg, torch.Tensor) and arg.is_complex()
        else arg
        for arg in model_args
    )
