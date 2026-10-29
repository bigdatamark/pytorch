# [MODULE DOC]
# Draft export: a lenient variant of export that always tries to produce an ExportedProgram
# while recording problems (missing or mismatched fake kernels, data dependent errors, guards
# added during tracing) from structured trace logs, and attaching a printable report to the
# program. Contains the failure/report classes, a structured trace capturing handler and the
# draft_export entry point. Import-time side effect: creates the module logger.
import getpass
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
import torch._logging._internal
import torch.utils._pytree as pytree
from torch._dynamo.exc import UserError, UserErrorType
from torch._export.passes.insert_custom_op_guards import (
    get_op_profiles,
    insert_custom_op_guards,
    OpProfile,
)
from torch._utils_internal import log_draft_export_usage

from ._trace import _export, get_ep_stats
from .dynamic_shapes import _DimHint, _DimHintType, Dim
from .exported_program import ExportedProgram


log = logging.getLogger(__name__)


_DRAFT_EXPORT_WARNING_COLOR = "\033[31m"
_DRAFT_EXPORT_SUCCESS_COLOR = "\033[32m"
_DRAFT_EXPORT_END_COLOR = "\033[0m"


def _format_colored(text: str, color: str) -> str:
    # [FUNCTION DOC] HELPER: wraps text in the given terminal color escape code followed by the reset code.
    return f"{color}{text}{_DRAFT_EXPORT_END_COLOR}"


class FailureType(IntEnum):
    # [CLASS DOC]
    # Integer enumeration of the kinds of problems draft export reports: missing fake kernel,
    # data dependent error, guard added, mismatched fake kernel. String form is the member name.
    MISSING_FAKE_KERNEL = 1
    DATA_DEPENDENT_ERROR = 2
    GUARD_ADDED = 3
    MISMATCHED_FAKE_KERNEL = 4

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the member name rather than the integer value.
        return self.name


def prettify_stack(stack: list[dict[str, str]], str_to_filename: dict[int, str]) -> str:
    # [FUNCTION DOC]
    # Formats a stack (list of frame records with filename id, line, name, loc) as indented text.
    # Frames whose filename id is not in str_to_filename are skipped; others print file, line
    # and function. Finally appends the source line text of the last frame.
    # Raises an error if the stack is empty (the last frame is read unconditionally).
    res = ""
    for frame in stack:
        if frame["filename"] not in str_to_filename:
            continue

        res += f"""
        File {str_to_filename[frame["filename"]]}, lineno {frame["line"]}, in {frame["name"]}"""  # type: ignore[index]

    res += f"\n            {stack[-1]['loc']}"
    return res


def prettify_frame_locals(
    loc: str, locals: dict[str, Any], symbols: dict[str, Any]
) -> str:
    # [FUNCTION DOC]
    # Formats captured frame locals as a "Locals:" text block, plus a "Symbols:" block listing
    # only symbols whose value is not None (omitted if all are None). Returns the string.
    # KNOWN ISSUE: the loc argument is accepted but never used.
    local_str = "\n".join(f"            {k}: {v}" for k, v in locals.items())
    res = f"""
        Locals:
{local_str}
"""
    if any(v is not None for v in symbols.values()):
        symbol_str = "\n".join(
            f"           {k}: {v}" for k, v in symbols.items() if v is not None
        )
        res += f"""
        Symbols:
{symbol_str}
"""
    return res


def get_loc(filename: str, lineno: int) -> str | None:
    # [FUNCTION DOC]
    # Reads the file and returns the stripped text of the one-based line number lineno, or None
    # if the file does not exist or has fewer lines. Performs file I/O (read only).
    try:
        with open(filename) as f:
            for i, line in enumerate(f):
                if i == lineno - 1:
                    return line.strip()
    except FileNotFoundError:
        pass
    return None


class FailureReport:
    # [CLASS DOC]
    # One reported problem. Holds the failure type, a data mapping with type-specific fields
    # (op name, reason, expression, stacks, frame locals, new dynamic shapes, ...) and an xfail
    # flag meaning the failure is expected and does not make the report unsuccessful.
    # Key method: print, which renders a human readable message.
    def __init__(
        self, failure_type: FailureType, data: dict[str, Any], xfail: bool = False
    ) -> None:
        # [FUNCTION DOC] Stores the failure type, data mapping and xfail flag.
        self.failure_type: FailureType = failure_type
        self.data: dict[str, Any] = data
        self.xfail: bool = xfail

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string showing failure type, xfail flag and data.
        return f"FailureReport(failure_type={self.failure_type}, xfail={self.xfail}, data={self.data})"

    def print(self, str_to_filename: dict[int, str]) -> str:
        # [FUNCTION DOC]
        # Builds the explanatory message for this failure type (despite the name, it returns the
        # text and does not print). Missing/mismatched fake kernel messages name the op (and reason)
        # with documentation links; guard-added and data-dependent messages include formatted user
        # and framework stack traces (via str_to_filename) and optional frame locals, plus the
        # suggested dynamic shapes or occurrence count. Raises ValueError for an unknown type.
        if self.failure_type == FailureType.MISSING_FAKE_KERNEL:
            op = self.data["op"]

            return f"""Missing fake kernel.
    torch.ops.{op} is missing a fake kernel implementation.

    Please refer to https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz for more detailed instructions on how to write a meta implementation.
"""

        elif self.failure_type == FailureType.GUARD_ADDED:
            locals_info = (
                prettify_frame_locals(**self.data["frame_locals"])
                if self.data["frame_locals"]
                else ""
            )
            return f"""Guard Added.
    A guard was added during tracing, which might've resulted in some incorrect
    tracing or constraint violation error.
    Specifically, this guard was added: {self.data["expr"]}, where {self.data["symbol_to_sources"]}.
    This occurred at the following stacktrace: {prettify_stack(self.data["user_stack"], str_to_filename)}:
        {locals_info}
    And the following framework stacktrace: {prettify_stack(self.data["stack"], str_to_filename)}\n
    Because of this, we have modified the dynamic shapes structure to be the
    following. You can also use torch.export.Dim.AUTO instead to specify your
    dynamic shapes, and we will automatically infer the dynamism for you.
    ```
    dynamic_shapes = {self.data["new_dynamic_shapes"]}
    ```
"""

        elif self.failure_type == FailureType.DATA_DEPENDENT_ERROR:
            locals_info = (
                prettify_frame_locals(**self.data["frame_locals"])
                if self.data["frame_locals"]
                else ""
            )
            return f"""Data dependent error.
    When exporting, we were unable to evaluate the value of `{self.data["expr"]}`.
    This was encountered {self.data["occurrences"]} times.
    This occurred at the following user stacktrace: {prettify_stack(self.data["user_stack"], str_to_filename)}
        {locals_info}
    And the following framework stacktrace: {prettify_stack(self.data["stack"], str_to_filename)}\n
    As a result, it was specialized to a constant (e.g. `{self.data["result"]}` in the 1st occurrence), and asserts were inserted into the graph.

    Please add `torch._check(...)` to the original code to assert this data-dependent assumption.
    Please refer to https://docs.google.com/document/d/1kZ_BbB3JnoLbUZleDT6635dHs88ZVYId8jT-yTFgf3A/edit#heading=h.boi2xurpqa0o for more details.
"""

        elif self.failure_type == FailureType.MISMATCHED_FAKE_KERNEL:
            op = self.data["op"]
            reason = self.data["reason"]
            return f"""Mismatched fake kernel.
    torch.ops.{op} has a fake kernel implementation, but it has incorrect behavior, based on the real kernel.
    The reason for the mismatch is: {reason}.

    Please refer to https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz for more detailed instructions on how to write a fake implementation.
"""

        else:
            raise ValueError(f"Unknown failure type: {self.failure_type}")


class DraftExportReport:
    # [CLASS DOC]
    # Result report of a draft export, stored on the exported program. Attributes: failures
    # (list of FailureReport), str_to_filename (id to filename table for stacks),
    # expressions_created (symbolic expression logs that were relevant), op_profiles (custom op
    # profiles for ops lacking fake kernels). Key methods: successful and string conversion.
    def __init__(
        self,
        failures: list[FailureReport],
        str_to_filename: dict[int, str],
        expressions_created: dict[int, dict[str, Any]],
        op_profiles: dict[str, set[OpProfile]],
    ):
        # [FUNCTION DOC] Stores the given failures, filename table, expression logs and op profiles.
        self.failures: list[FailureReport] = failures
        self.str_to_filename = str_to_filename
        self.expressions_created: dict[int, dict[str, Any]] = expressions_created
        self.op_profiles = op_profiles

    def successful(self) -> bool:
        # [FUNCTION DOC] True if there are no failures or every failure is marked xfail.
        return len(self.failures) == 0 or all(
            failure.xfail for failure in self.failures
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string showing the failure list.
        return f"DraftExportReport({self.failures})"

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns a green banner if successful; otherwise a red warning banner with the
        # failure count followed by a numbered list of each failure's message.
        if self.successful():
            return _format_colored(
                """
##############################################################################################
Congratulations: No issues are found during export, and it was able to soundly produce a graph.
You can now change back to torch.export.export()
##############################################################################################
""",
                _DRAFT_EXPORT_SUCCESS_COLOR,
            )

        error = _format_colored(
            f"""
###################################################################################################
WARNING: {len(self.failures)} issue(s) found during export, and it was not able to soundly produce a graph.
Please follow the instructions to fix the errors.
###################################################################################################

""",
            _DRAFT_EXPORT_WARNING_COLOR,
        )
        error += "\n"

        for i, failure in enumerate(self.failures):
            error += f"{i + 1}. {failure.print(self.str_to_filename)}\n"
        return error

    def apply_suggested_fixes(self) -> None:
        # [FUNCTION DOC] Placeholder; always raises NotImplementedError.
        raise NotImplementedError("Not implemented yet")


@dataclass
class ExpressionCreatedNode:
    # [CLASS DOC]
    # Dataclass node for one logged symbolic expression creation: result_id, ids of argument
    # expressions, the original log record, and a visited flag used to emit each at most once
    # when walking dependencies.
    result_id: int
    argument_ids: list[int]
    record: dict[str, object]
    visited: bool = False


class LogRecord:
    # [CLASS DOC]
    # De-duplicating store of structured log entries (key, data). Attributes: log_count maps an
    # entry hash to its occurrence count; logs holds the first occurrence of each distinct
    # entry in order. Hash depends on the entry key (op, reason or user stack where relevant).
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes empty count mapping and empty log list.
        self.log_count: dict[int, int] = {}
        self.logs: list[tuple[str, dict[str, Any]]] = []

    def _hash(self, element: tuple[str, dict[str, Any]]) -> int:
        # [FUNCTION DOC] HELPER: computes the identity hash of a (key, data) entry. Uses the op for missing
        # fake kernels, op plus reason for mismatches, the serialized user stack for provenance,
        # guard and unbacked symbol entries, and the whole serialized data otherwise.
        key, data = element

        if key == "missing_fake_kernel":
            return hash((key, data["op"]))
        elif key == "mismatched_fake_kernel":
            return hash((key, data["op"], data["reason"]))
        elif key == "propagate_real_tensors_provenance":
            return hash((key, json.dumps(data["user_stack"])))
        elif key == "guard_added":
            return hash((key, json.dumps(data["user_stack"])))
        elif key == "create_unbacked_symbol":
            return hash((key, json.dumps(data["user_stack"])))

        return hash((key, json.dumps(data)))

    def try_add(self, element: tuple[str, dict[str, str]]) -> bool:
        # [FUNCTION DOC] Adds the entry if its hash is new (count 1, appended to logs) and returns True;
        # if seen before, increments its count and returns False.
        hash_value = self._hash(element)
        if hash_value in self.log_count:
            self.log_count[hash_value] += 1
            return False

        self.log_count[hash_value] = 1
        self.logs.append(element)
        return True

    def get_log_count(self, element: tuple[str, dict[str, Any]]) -> int:
        # [FUNCTION DOC] Returns how many times an equivalent entry was added; raises KeyError if never added.
        return self.log_count[self._hash(element)]


class CaptureStructuredTrace(torch._logging._internal.LazyTraceHandler):
    # [CLASS DOC]
    # Logging handler and context manager that captures structured trace records relevant to
    # draft export while still writing them to a trace file. State: a LogRecord of unique
    # relevant entries, expression_created_logs keyed by result id, the list of recognized log
    # keys, the trace logger and the saved previous structured-tracing flag. On enter it
    # replaces lazy trace handlers on the trace logger and enables structured tracing; on exit
    # it reverses that state change.
    def __init__(self) -> None:
        # [FUNCTION DOC] Sets up state and initializes the base handler with a root directory taken from the
        # trace environment variable, or a per-user directory under the system temp dir (user name
        # sanitized). Installs the trace formatter.
        self.specific_log_keys = [
            "str",
            "exported_program",
            "propagate_real_tensors_provenance",
            "guard_added",
            "missing_fake_kernel",
            "mismatched_fake_kernel",
            "expression_created",
            "create_unbacked_symbol",
        ]
        self.log_record: LogRecord = LogRecord()
        self.expression_created_logs: dict[int, ExpressionCreatedNode] = {}
        self.symbol_to_expressions: dict[str, list[dict[str, Any]]] = {}
        self.logger = logging.getLogger("torch.__trace")
        self.prev_get_dtrace = False

        if root_dir := os.environ.get(torch._logging._internal.DTRACE_ENV_VAR):
            super().__init__(root_dir)
        else:
            sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
            root_dir = os.path.join(
                tempfile.gettempdir(),
                "export_" + sanitized_username,
            )
            super().__init__(root_dir)

        self.setFormatter(torch._logging._internal.TorchLogsFormatter(trace=True))

    def __enter__(self) -> "CaptureStructuredTrace":
        # [FUNCTION DOC] Resets captured state, removes any lazy trace handlers from the trace logger, adds
        # this handler, saves and sets the global structured-trace flag to true. Returns self.
        # KNOWN ISSUE: removed lazy handlers are not re-added on exit.
        self.log_record = LogRecord()
        self.expression_created_logs = {}

        # Remove the lazy trace handler if it exists
        possible_lazy_trace_handlers = [
            handler
            for handler in self.logger.handlers
            if isinstance(handler, torch._logging._internal.LazyTraceHandler)
        ]
        for handler in possible_lazy_trace_handlers:
            self.logger.removeHandler(handler)

        self.logger.addHandler(self)
        self.prev_get_dtrace = torch._logging._internal.GET_DTRACE_STRUCTURED
        # pyrefly: ignore [bad-assignment]
        torch._logging._internal.GET_DTRACE_STRUCTURED = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Clears captured state, detaches this handler from the logger and restores the
        # previous global structured-trace flag. Does not suppress exceptions.
        self.log_record = LogRecord()
        self.expression_created_logs = {}
        self.logger.removeHandler(self)
        # pyrefly: ignore [bad-assignment]
        torch._logging._internal.GET_DTRACE_STRUCTURED = self.prev_get_dtrace
        self.prev_get_dtrace = False

    def emit(self, record: Any) -> None:
        # [FUNCTION DOC]
        # Handler hook for each log record. For each recognized key present in the record
        # metadata, keeps it only if new (de-duplicated). expression_created records are stored
        # by result id and not emitted. Provenance and guard records emit their dependent
        # expression records first; guard records with no input sources or runtime-assert
        # prefix are dropped. Remaining kept records are written via the base handler.
        def _log_expression_created(
            emit_func: Callable[[Any], None], sym_node_id: int
        ) -> None:
            # [FUNCTION DOC] HELPER: recursively emits the stored expression record for sym_node_id, first
            # emitting (once, tracked by the visited flag) the records of its argument ids. Does
            # nothing for a missing id.
            # KNOWN ISSUE: the record itself is emitted again on every call even when visited.
            # Log all the relevant expression_created logs
            if sym_node_id is None:
                return
            if res := self.expression_created_logs.get(sym_node_id, None):
                # Don't log the expression if we have already
                # printed it beforehand
                if not res.visited:
                    res.visited = True
                    for arg in res.argument_ids:
                        _log_expression_created(emit_func, arg)

                emit_func(res.record)

        metadata = record.metadata
        for key in self.specific_log_keys:
            if key in metadata:
                if self.log_record.try_add((key, metadata[key])):
                    if key == "expression_created":
                        # We don't want to log all expression_created logs, only
                        # the ones that are relevant to the
                        # guards/propagate_real_tensor
                        self.expression_created_logs[metadata[key]["result_id"]] = (
                            ExpressionCreatedNode(
                                metadata[key]["result_id"],
                                metadata[key].get("argument_ids", []),
                                record,
                            )
                        )
                        return

                    elif key == "propagate_real_tensors_provenance":
                        _log_expression_created(
                            super().emit, metadata[key].get("expr_node_id")
                        )

                    elif key == "guard_added":
                        if len(metadata[key]["symbol_to_sources"]) == 0:
                            # We only want to include guards added that are relevant to
                            # the symbolic shapes corresponding to the inputs which were
                            # specified in the dynamic_shapes arg. These have a source.
                            return
                        elif metadata[key]["prefix"] == "runtime_assert":
                            # This should've been captured by a
                            # propagate_real_tensors log
                            return

                        _log_expression_created(
                            super().emit, metadata[key].get("expr_node_id")
                        )

                    super().emit(record)


def draft_export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None = None,
    *,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None,
    preserve_module_call_signature: tuple[str, ...] = (),
    strict: bool = False,
    pre_dispatch: bool = True,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
) -> ExportedProgram:
    # [FUNCTION DOC]
    # Exports mod with the given args/kwargs like export, but tolerates problems and collects
    # them. Runs the internal export with fake-tensor real-propagation and generated fake kernels
    # enabled while capturing structured traces. On a constraint violation error, retries with
    # all dynamic dims converted to automatic; any other error is logged as usage and re-raised.
    # Converts captured logs into FailureReports, builds a DraftExportReport (with op profiles
    # for ops with missing/mismatched fake kernels), inserts guards around those custom ops, and
    # attaches the report to the program as its report attribute. Logs a warning with a tlparse
    # hint if unsuccessful, else an info message; logs usage. Returns the ExportedProgram.
    # KNOWN ISSUE: annotated dynamic_shapes tuple type is a one-element tuple.
    start_time = time.time()
    kwargs = kwargs or {}
    dynamic_shapes = dynamic_shapes or {}

    constraint_violation_msg = None
    capture_structured_log = CaptureStructuredTrace()

    with (
        torch._functorch.config.patch(
            fake_tensor_propagate_real_tensors=True,
            generate_fake_kernels_from_real_mismatches=True,
        ),
        capture_structured_log,
    ):
        try:
            new_shapes = None
            ep = _export(
                mod,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
                pre_dispatch=pre_dispatch,
                preserve_module_call_signature=preserve_module_call_signature,
                prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
            )
        except Exception as exc:
            if (
                isinstance(exc, UserError)
                and exc.error_type == UserErrorType.CONSTRAINT_VIOLATION
            ):
                constraint_violation_msg = exc.msg

                def convert_dim_to_auto(dim: Any) -> Any:
                    # [FUNCTION DOC] HELPER: converts a Dim or a dynamic dim hint into an automatic dim keeping
                    # min and max; any other value is returned unchanged.
                    if isinstance(dim, Dim):
                        return Dim.AUTO(min=dim.min, max=dim.max)
                    elif isinstance(dim, _DimHint) and dim.type == _DimHintType.DYNAMIC:
                        return Dim.AUTO(min=dim.min, max=dim.max)
                    return dim

                new_shapes = pytree.tree_map(convert_dim_to_auto, dynamic_shapes)
                ep = _export(
                    mod,
                    args,
                    kwargs,
                    dynamic_shapes=new_shapes,
                    strict=strict,
                    pre_dispatch=pre_dispatch,
                    preserve_module_call_signature=preserve_module_call_signature,
                    prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
                )
            else:
                log_draft_export_usage(
                    error=True,
                    export_time=time.time() - start_time,
                    strict=strict,
                    message=str(exc),
                    type=f"{type(exc).__name__}.{type(exc).__qualname__}",
                )
                raise exc

        torch._logging.dtrace_structured("exported_program", payload_fn=lambda: str(ep))

        str_to_filename: dict[int, str] = {}
        failures: list[FailureReport] = []
        incorrect_custom_ops: set[str] = set()
        expressions_created: dict[int, dict[str, Any]] = {}

        for log_name, log_contents in capture_structured_log.log_record.logs:
            failure_type = None

            if log_name == "str":
                str_to_filename[log_contents[1]] = log_contents[0]  # type: ignore[index]
                continue

            elif log_name == "propagate_real_tensors_provenance":
                log_contents["occurrences"] = (
                    capture_structured_log.log_record.get_log_count(
                        (log_name, log_contents)
                    )
                )

                failure_type = FailureType.DATA_DEPENDENT_ERROR

            elif log_name == "guard_added":
                if new_shapes is None:
                    continue

                failure_type = FailureType.GUARD_ADDED
                log_contents["new_dynamic_shapes"] = new_shapes
            elif log_name == "missing_fake_kernel":
                failure_type = FailureType.MISSING_FAKE_KERNEL
                incorrect_custom_ops.add(log_contents["op"])

            elif log_name == "mismatched_fake_kernel":
                failure_type = FailureType.MISMATCHED_FAKE_KERNEL
                incorrect_custom_ops.add(log_contents["op"])

            else:
                continue

            if failure_type is None:
                raise AssertionError("failure_type cannot be None at this point")
            failures.append(
                FailureReport(
                    failure_type,
                    log_contents,
                )
            )

        for k, v in capture_structured_log.expression_created_logs.items():
            if v.visited:
                expressions_created[k] = v.record

        op_profiles = get_op_profiles(ep.graph_module, incorrect_custom_ops)
        report = DraftExportReport(
            failures, str_to_filename, expressions_created, op_profiles
        )

        # Add asserts around custom ops
        insert_custom_op_guards(ep.graph_module, incorrect_custom_ops)

    ep._report = report
    if not report.successful():
        log_filename = capture_structured_log.stream.name

        warning_msg = f"""
###################################################################################################
WARNING: {len(report.failures)} issue(s) found during export, and it was not able to soundly produce a graph.
To view the report of failures in an html page, please run the command:
    `tlparse {log_filename} --export`
Or, you can view the errors in python by inspecting `print(ep._report)`.
"""

        if len(report.op_profiles) > 0:
            warning_msg += f"""
While tracing we found {len(report.op_profiles)} operator(s) which do not have a fake kernel registered.
If you intend to retrace the exported graph or run it with fake tensors, please run it under the
following context manager, which will register a fake kernel for those operators.
```
with torch._library.fake_profile.unsafe_generate_fake_kernels(ep._report.op_profiles):
    # run with fake tensors
```
"""

        warning_msg += """#################################################################################################"""

        log.warning(warning_msg)

    else:
        log.info(
            """
##############################################################################################
Congratulations: No issues are found during export, and it was able to soundly produce a graph.
You can now change back to torch.export.export()
##############################################################################################
    """
        )

    log_draft_export_usage(
        error=False,
        export_time=time.time() - start_time,
        strict=strict,
        constraint_violations=constraint_violation_msg,
        report=ep._report,
        **get_ep_stats(ep),
    )
    return ep
