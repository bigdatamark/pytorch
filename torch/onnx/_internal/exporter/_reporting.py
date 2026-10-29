# mypy: allow-untyped-defs
# [MODULE DOC] Builds the markdown reports written when ONNX export fails or when reporting is
# requested: an ExportStatus record of which stages succeeded, helpers to format that status,
# the exported program, the decomposition comparison and the verification results, a file name
# builder, and two functions that write the reports. No import-time side effects.
from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

from torch.onnx._internal.exporter import _analysis, _registration, _verification


if TYPE_CHECKING:
    import os

    import onnx_ir as ir  # type: ignore[import-untyped]

    import torch


@dataclasses.dataclass
class ExportStatus:
    # [CLASS DOC] Dataclass of tri-state flags (true, false, or none for not run) recording
    # which export stages succeeded: strict, non-strict and draft torch.export,
    # decomposition, ONNX translation, onnx checker, ONNX Runtime execution, and output
    # accuracy. All default to none.
    # Whether torch.export.export(..., strict=True) succeeds
    torch_export_strict: bool | None = None
    # Whether torch.export.export(..., strict=False) succeeds
    torch_export_non_strict: bool | None = None
    # Whether torch.export.draft_export() succeeds
    torch_export_draft_export: bool | None = None
    # Whether decomposition succeeds
    decomposition: bool | None = None
    # Whether ONNX translation succeeds
    onnx_translation: bool | None = None
    # Whether ONNX model passes onnx.checker.check_model
    onnx_checker: bool | None = None
    # Whether ONNX model runs successfully with ONNX Runtime
    onnx_runtime: bool | None = None
    # Whether the output of the ONNX model is accurate
    output_accuracy: bool | None = None


def _status_emoji(status: bool | None) -> str:
    # [FUNCTION DOC] HELPER: Maps a tri-state flag to a symbol: white circle for none, check
    # mark for true, cross mark for false.
    if status is None:
        return "⚪"
    return "✅" if status else "❌"


def _format_export_status(status: ExportStatus) -> str:
    # [FUNCTION DOC] HELPER: Returns a fenced markdown block with one line per export stage,
    # each prefixed by its status symbol, followed by blank lines.
    return (
        f"```\n"
        f"{_status_emoji(status.torch_export_non_strict)} Obtain model graph with `torch.export.export(..., strict=False)`\n"
        f"{_status_emoji(status.torch_export_strict)} Obtain model graph with `torch.export.export(..., strict=True)`\n"
        f"{_status_emoji(status.torch_export_draft_export)} Obtain model graph with `torch.export.draft_export`\n"
        f"{_status_emoji(status.decomposition)} Decompose operators for ONNX compatibility\n"
        f"{_status_emoji(status.onnx_translation)} Translate the graph into ONNX\n"
        f"{_status_emoji(status.onnx_checker)} Run `onnx.checker` on the ONNX model\n"
        f"{_status_emoji(status.onnx_runtime)} Execute the model with ONNX Runtime\n"
        f"{_status_emoji(status.output_accuracy)} Validate model output accuracy\n"
        f"```\n\n"
    )


def _strip_color_from_string(text: str) -> str:
    # [FUNCTION DOC] HELPER: Removes ANSI escape sequences (terminal colors) from text.
    # This regular expression matches ANSI escape codes
    # https://github.com/pytorch/pytorch/blob/9554a9af8788c57e1c5222c39076a5afcf0998ae/torch/_dynamo/utils.py#L2785-L2788
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def _format_exported_program(exported_program: torch.export.ExportedProgram) -> str:
    # [FUNCTION DOC] HELPER: Returns the printed exported program, without colors, in a
    # fenced python markdown block. Uses the string form because a no-color print option may
    # be missing in older versions.
    # Adapted from https://github.com/pytorch/pytorch/pull/128476
    # to remove colors
    # Even though we can call graph_module.print_readable directly, since the
    # colored option was added only recently, we can't guarantee that the
    # version of PyTorch used by the user has this option. Therefore, we
    # still call str(ExportedProgram)
    text = f"```python\n{_strip_color_from_string(str(exported_program))}\n```\n\n"
    return text


def construct_report_file_name(timestamp: str, status: ExportStatus) -> str:
    # [FUNCTION DOC] Builds the report file name from the timestamp and the first failing
    # stage: pt_export if no torch.export strategy succeeded, then decomp, conversion,
    # checker, runtime, accuracy, strategies (some strategy failed but a later stage
    # worked), else success. Only an explicit false counts as failure for the later stages,
    # none does not.
    # KNOWN ISSUE: when no strategy was run at all (all none) the name still ends with
    # pt_export.
    # Status could be None. So we need to check for False explicitly.
    if not (
        status.torch_export_non_strict
        or status.torch_export_strict
        or status.torch_export_draft_export
    ):
        # All strategies failed
        postfix = "pt_export"
    elif status.decomposition is False:
        postfix = "decomp"
    elif status.onnx_translation is False:
        postfix = "conversion"
    elif status.onnx_checker is False:
        postfix = "checker"
    elif status.onnx_runtime is False:
        postfix = "runtime"
    elif status.output_accuracy is False:
        postfix = "accuracy"
    elif (
        status.torch_export_strict is False
        or status.torch_export_non_strict is False
        or status.torch_export_draft_export is False
    ):
        # Some strategies failed
        postfix = "strategies"
    else:
        postfix = "success"
    return f"onnx_export_{timestamp}_{postfix}.md"


def format_decomp_comparison(
    pre_decomp_unique_ops: set[str],
    post_decomp_unique_ops: set[str],
) -> str:
    # [FUNCTION DOC] Formats the sets of operators found only before and only after
    # decomposition as two sorted lists in markdown text. Args: pre_decomp_unique_ops,
    # post_decomp_unique_ops.
    """Format the decomposition comparison result.

    Args:
        pre_decomp_unique_ops: The unique ops in the first program.
        post_decomp_unique_ops: The unique ops in the second program.

    Returns:
        The formatted comparison result.
    """
    return (
        f"Ops exist only in the ExportedProgram before decomposition: `{sorted(pre_decomp_unique_ops)}`\n\n"
        f"Ops exist only in the ExportedProgram after decomposition: `{sorted(post_decomp_unique_ops)}`\n"
    )


def format_verification_infos(
    verification_infos: list[_verification.VerificationInfo],
) -> str:
    # [FUNCTION DOC] Formats one line per verification result with its name, maximum
    # absolute and relative differences and the difference histograms; lines are joined by
    # newlines.
    """Format the verification result.

    Args:
        verification_infos: The verification result.

    Returns:
        The formatted verification result.
    """
    return "\n".join(
        f"`{info.name}`: `max_abs_diff={info.max_abs_diff:e}`, `max_rel_diff={info.max_rel_diff:e}`, "
        f"`abs_diff_hist={info.abs_diff_hist}`, `rel_diff_hist={info.rel_diff_hist}`"
        for info in verification_infos
    )


def create_torch_export_error_report(
    filename: str | os.PathLike,
    formatted_traceback: str,
    *,
    export_status: ExportStatus,
    profile_result: str | None,
) -> None:
    # [FUNCTION DOC] Writes a markdown error report to filename (overwriting it, UTF-8): a
    # title, the status block, the formatted traceback, and the profiling result if given.
    # Args: filename, formatted_traceback, export_status, profile_result. Side effect: file
    # I/O.
    # KNOWN ISSUE: the traceback is written without a newline before the closing code fence,
    # unlike the other report writer, so the fence may be joined to the last traceback line.
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# PyTorch ONNX Conversion Error Report\n\n")
        f.write(_format_export_status(export_status))
        f.write("Error message:\n\n")
        f.write("```pytb\n")
        f.write(formatted_traceback)
        f.write("```\n\n")
        if profile_result is not None:
            f.write("## Profiling result\n\n")
            f.write("```\n")
            f.write(profile_result)
            f.write("```\n")


def create_onnx_export_report(
    filename: str | os.PathLike,
    formatted_traceback: str,
    program: torch.export.ExportedProgram,
    *,
    decomp_comparison: str | None = None,
    export_status: ExportStatus,
    profile_result: str | None,
    model: ir.Model | None = None,
    registry: _registration.ONNXRegistry | None = None,
    verification_result: str | None = None,
) -> None:
    # [FUNCTION DOC] Writes the full markdown conversion report to filename (overwriting it,
    # UTF-8): status block, traceback, the exported program, the ONNX model if given, an
    # operator analysis of the program (computed with the registry), and optional sections
    # for the decomposition comparison, verification results and profiling result. Args:
    # filename, formatted_traceback, program, decomp_comparison, export_status,
    # profile_result, model, registry, verification_result. Side effects: file I/O and the
    # analysis writes to the same file.
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# PyTorch ONNX Conversion Report\n\n")
        f.write(_format_export_status(export_status))
        f.write("## Error messages\n\n")
        f.write("```pytb\n")
        f.write(formatted_traceback)
        f.write("\n```\n\n")
        f.write("## Exported program\n\n")
        f.write(_format_exported_program(program))
        if model is not None:
            f.write("## ONNX model\n\n")
            f.write("```python\n")
            f.write(str(model))
            f.write("\n```\n\n")
        f.write("## Analysis\n\n")
        _analysis.analyze(program, file=f, registry=registry)
        if decomp_comparison is not None:
            f.write("\n## Decomposition comparison\n\n")
            f.write(decomp_comparison)
            f.write("\n")
        if verification_result is not None:
            f.write("\n## Verification results\n\n")
            f.write(verification_result)
            f.write("\n")
        if profile_result is not None:
            f.write("\n## Profiling result\n\n")
            f.write("```\n")
            f.write(profile_result)
            f.write("```\n")
