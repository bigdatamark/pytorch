"""Strategies for capturing ExportedPrograms."""

# mypy: allow-untyped-defs
# [MODULE DOC] Strategies for capturing a model as a torch.export ExportedProgram for the ONNX
# exporter: a base class with progress reporting hooks, strict, non-strict and draft-export
# strategies, patching helpers, and the CAPTURE_STRATEGIES tuple of strategies to try in order
# (draft export is appended only when the ENABLE_DRAFT_EXPORT flag is set, evaluated at import).
from __future__ import annotations

import abc
import contextlib
import dataclasses
import datetime
import logging
import pathlib
from typing import Any, TYPE_CHECKING

import torch
from torch.onnx import _flags


if TYPE_CHECKING:
    import os
    from collections.abc import Callable


logger = logging.getLogger(__name__)


def _verbose_printer(verbose: bool | None) -> Callable[..., None]:
    # [FUNCTION DOC] HELPER: Returns a print function. If verbose is exactly false, the function
    # discards its arguments; otherwise it prints the arguments to stdout prefixed with a
    # torch.onnx tag (so None also means verbose).
    """Prints messages based on `verbose`."""
    if verbose is False:
        return lambda *_, **__: None

    return lambda *args, **kwargs: print("[torch.onnx]", *args, **kwargs)


def _take_first_line(text: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the first line of the text, with a truncation marker
    # appended when the text has more than one line.
    """Take the first line of a text."""
    lines = text.split("\n", maxsplit=1)
    first_line = lines[0]
    if len(lines) > 1:
        first_line += "[...]"
    return first_line


@contextlib.contextmanager
def _patch_dynamo_unsupported_functions():
    # [FUNCTION DOC] HELPER: Context manager that temporarily replaces the torch.jit isinstance
    # function with the builtin isinstance (global mutation) so dynamo can trace, logs at info
    # level, and restores the original on exit even if the body raises.
    """Patch PyTorch to bypass some functions torch.export.export does not support."""
    # TODO: Remove the patches once dynamo supports these functions.
    import torch.jit

    # Replace torch.jit.isinstance with isinstance
    jit_isinstance = torch.jit.isinstance
    # pyrefly: ignore [bad-assignment]
    torch.jit.isinstance = isinstance
    logger.info("Replaced torch.jit.isinstance with isinstance to allow dynamo tracing")
    try:
        yield
    finally:
        torch.jit.isinstance = jit_isinstance


@contextlib.contextmanager
def _patch_dynamic_shape_rnn_decompositions(dynamic_shapes):
    # [FUNCTION DOC] HELPER: Context manager. If dynamic_shapes is empty it does nothing;
    # otherwise it lazily imports and enters the LSTM and GRU while-loop decomposition
    # registrations for the duration of the body.
    if not dynamic_shapes:
        yield
        return

    # Import lazily to avoid loading decomposition registrations on torch.onnx import.
    from torch.export._patches import (
        register_gru_while_loop_decomposition,
        register_lstm_while_loop_decomposition,
    )

    with (
        register_lstm_while_loop_decomposition(),
        register_gru_while_loop_decomposition(),
    ):
        yield


@dataclasses.dataclass
class Result:
    # [CLASS DOC] Dataclass holding the outcome of a capture attempt: the exported program (or
    # none on failure), the name of the strategy class that produced it, and an optional
    # exception (a failure cause, or informational for a successful draft export).
    exported_program: torch.export.ExportedProgram | None
    strategy: str
    exception: Exception | None = None

    @property
    def success(self) -> bool:
        # [FUNCTION DOC] Property: true exactly when an exported program is present; the
        # recorded exception is ignored.
        """Whether the capture was successful.

        An exception can still be recorded even if the capture was successful. In
        this case the exception is informational only. For example, draft_export
        can record an exception if there are warnings during the export. The exceptions
        will go into the onnx export report when report=True.
        """
        return self.exported_program is not None


class CaptureStrategy(abc.ABC):
    # [CLASS DOC] Abstract base for capture strategies. Instances are callable: they run the
    # subclass _capture, never raise on capture errors (returning a failed Result instead), and
    # call the _enter, _success and _failure hooks for progress reporting. State: a verbose
    # print function, dump flag, artifacts directory, a timestamp string, and an optional
    # informational exception that subclasses may set during capture.
    """Strategy for capturing a module as ExportedProgram.

    To use a strategy, create an instance and call it with the model, args, kwargs, and dynamic_shapes.
    Example::

        strategy = TorchExportNonStrictStrategy(verbose=True)
        result = strategy(model, args, kwargs, dynamic_shapes)
    """

    def __init__(
        self,
        *,
        verbose: bool = False,
        dump: bool = False,
        artifacts_dir: str | os.PathLike = ".",
        timestamp: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the verbose printer, dump flag, artifacts directory (as a path)
        # and timestamp (defaults to the current time formatted with microseconds), and clears
        # the stored exception. KNOWN ISSUE: docstring omits artifacts_dir and timestamp args.
        """Initialize the strategy.

        Args:
            verbose: Whether to print verbose messages.
            dump: Whether to dump the intermediate artifacts to a file.
        """
        self._verbose_print = _verbose_printer(verbose)
        self._dump = dump
        self._artifacts_dir = pathlib.Path(artifacts_dir)
        self._timestamp = timestamp or datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S-%f"
        )
        self._exception: Exception | None = None

    def __call__(
        self,
        model: torch.nn.Module | torch.jit.ScriptFunction,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
        dynamic_shapes,
    ) -> Result:
        # [FUNCTION DOC] Runs the capture: calls the enter hook, treats missing kwargs as an
        # empty mapping, and calls _capture. On any Exception calls the failure hook and
        # returns a Result with no program and that exception; otherwise calls the success hook
        # and returns a Result with the program and any informational stored exception.
        self._enter(model)
        if kwargs is None:
            kwargs = {}
        try:
            exported_program = self._capture(model, args, kwargs, dynamic_shapes)
        except Exception as e:
            self._failure(model, e)
            return Result(
                exported_program=None,
                strategy=self.__class__.__name__,
                exception=e,
            )
        self._success(model)
        return Result(
            exported_program,
            strategy=self.__class__.__name__,
            exception=self._exception,
        )

    @abc.abstractmethod
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        # [FUNCTION DOC] HELPER: Abstract hook; subclasses capture the model with the given
        # args, kwargs and dynamic shapes and return an ExportedProgram, raising on failure.
        raise NotImplementedError

    def _enter(self, model: torch.nn.Module | torch.jit.ScriptFunction) -> None:
        # [FUNCTION DOC] HELPER: Hook called before capture; default does nothing.
        return

    def _success(self, model: torch.nn.Module | torch.jit.ScriptFunction) -> None:
        # [FUNCTION DOC] HELPER: Hook called after a successful capture; default does nothing.
        return

    def _failure(
        self, model: torch.nn.Module | torch.jit.ScriptFunction, e: Exception
    ) -> None:
        # [FUNCTION DOC] HELPER: Hook called after a failed capture; default does nothing.
        return


class TorchExportStrictStrategy(CaptureStrategy):
    # [CLASS DOC] Capture strategy using torch.export.export in strict (dynamo tracing) mode.
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        # [FUNCTION DOC] Exports the model in strict mode with the jit isinstance patch, RNN
        # while-loop decompositions (when dynamic shapes are given) and backed size oblivious
        # mode active. On a dynamo user error, tries to refine dynamic shapes from the suggested
        # fixes and re-exports once; if refinement fails the original error is re-raised.
        with (
            _patch_dynamo_unsupported_functions(),
            _patch_dynamic_shape_rnn_decompositions(dynamic_shapes),
            # Support the dynamism with 0/1 input dim
            torch.fx.experimental._config.patch(backed_size_oblivious=True),  # type: ignore[attr-defined]
        ):
            try:
                return torch.export.export(
                    model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=True,
                    prefer_deferred_runtime_asserts_over_guards=_flags.PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS,
                )
            except torch._dynamo.exc.UserError as exc:
                # Refine the dynamic shapes based on the suggested fixes.
                try:
                    new_shapes = torch.export.dynamic_shapes.refine_dynamic_shapes_from_suggested_fixes(
                        exc.msg, dynamic_shapes
                    )
                except Exception:
                    # If the dynamic shapes cannot be refined, re-raise the exception.
                    raise exc from None
                return torch.export.export(
                    model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=new_shapes,
                    strict=True,
                    prefer_deferred_runtime_asserts_over_guards=_flags.PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS,
                )

    def _enter(self, model) -> None:
        # [FUNCTION DOC] HELPER: Prints a progress message naming the model's first repr line.
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=True)`..."
        )

    def _success(self, model) -> None:
        # [FUNCTION DOC] HELPER: Prints the progress message with a success mark.
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=True)`... ✅"
        )

    def _failure(self, model, e) -> None:
        # [FUNCTION DOC] HELPER: Prints the progress message with a failure mark; ignores e.
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=True)`... ❌"
        )


class TorchExportNonStrictStrategy(CaptureStrategy):
    # [CLASS DOC] Capture strategy using torch.export.export in non-strict mode (preferred).
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        # [FUNCTION DOC] Same as the strict strategy but with strict mode off and without the
        # jit isinstance patch: exports with RNN decompositions and backed size oblivious mode,
        # and on a dynamo user error retries once with dynamic shapes refined from suggested
        # fixes (re-raising the original error if refinement fails).
        with (
            _patch_dynamic_shape_rnn_decompositions(dynamic_shapes),
            # Support the dynamism with 0/1 input dim
            torch.fx.experimental._config.patch(backed_size_oblivious=True),  # type: ignore[attr-defined]
        ):
            try:
                return torch.export.export(
                    model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                    prefer_deferred_runtime_asserts_over_guards=_flags.PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS,
                )
            except torch._dynamo.exc.UserError as exc:
                # Refine the dynamic shapes based on the suggested fixes.
                try:
                    new_shapes = torch.export.dynamic_shapes.refine_dynamic_shapes_from_suggested_fixes(
                        exc.msg, dynamic_shapes
                    )
                except Exception:
                    # If the dynamic shapes cannot be refined, re-raise the exception.
                    raise exc from None
                return torch.export.export(
                    model,
                    args,
                    kwargs=kwargs,
                    dynamic_shapes=new_shapes,
                    strict=False,
                    prefer_deferred_runtime_asserts_over_guards=_flags.PREFER_DEFERRED_RUNTIME_ASSERTS_OVER_GUARDS,
                )

    def _enter(self, model) -> None:
        # [FUNCTION DOC] HELPER: Prints a progress message naming the model's first repr line.
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`..."
        )

    def _success(self, model) -> None:
        # [FUNCTION DOC] HELPER: Prints the progress message with a success mark.
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`... ✅"
        )

    def _failure(self, model, e) -> None:
        # [FUNCTION DOC] HELPER: Prints the progress message with a failure mark; ignores e.
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.export(..., strict=False)`... ❌"
        )


class TorchExportDraftExportStrategy(CaptureStrategy):
    # [CLASS DOC] Capture strategy using draft_export, which tolerates some export problems and
    # records them in a report; used only when the draft export flag is enabled.
    def _capture(
        self, model, args, kwargs, dynamic_shapes
    ) -> torch.export.ExportedProgram:
        # [FUNCTION DOC] Runs draft export and returns the program. If its report is not
        # successful, stores a RuntimeError containing the report as the informational exception
        # and prints the report; the capture still counts as a success.
        ep = torch.export.draft_export(
            model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
        )
        report = ep._report  # type: ignore[attr-defined]
        if not report.successful():
            self._exception = RuntimeError(str(report))
            self._verbose_print(f"Draft Export report:\n{report}")
        return ep

    def _enter(self, model) -> None:
        # [FUNCTION DOC] HELPER: Prints a progress message naming the model's first repr line.
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.draft_export`..."
        )

    def _success(self, model) -> None:
        # [FUNCTION DOC] HELPER: Prints the progress message with a success mark.
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.draft_export`... ✅"
        )

    def _failure(self, model, e) -> None:
        # [FUNCTION DOC] HELPER: Prints the progress message with a failure mark; ignores e.
        del e  # Unused
        model_repr = _take_first_line(repr(model))
        self._verbose_print(
            f"Obtain model graph for `{model_repr}` with `torch.export.draft_export`... ❌"
        )


CAPTURE_STRATEGIES: tuple[type[CaptureStrategy], ...] = (
    TorchExportNonStrictStrategy,  # strict=False is preferred over strict=True because it does not have dynamo issues
    TorchExportStrictStrategy,
)

if _flags.ENABLE_DRAFT_EXPORT:
    CAPTURE_STRATEGIES = (*CAPTURE_STRATEGIES, TorchExportDraftExportStrategy)
