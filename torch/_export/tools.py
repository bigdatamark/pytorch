# [MODULE DOC] Export diagnostics tooling. Public API: report_exportability, which tries to export a
# module and, on failure, each of its submodules, to localize export problems.
# Import-time side effects: creates a module logger only.
# mypy: allow-untyped-defs
import logging
import warnings
from collections.abc import Iterable
from typing import Any

import torch
import torch.export
import torch.export._trace
from torch._utils_internal import log_export_usage


log = logging.getLogger(__name__)

__all__ = ["report_exportability"]


def _generate_inputs_for_submodules(
    model: torch.nn.Module,
    target_submodules: Iterable[str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> dict[str, tuple[Any, Any]]:
    # [FUNCTION DOC] HELPER: Run the model once on the given inputs with forward pre-hooks registered on the
    # named target submodules, recording each one's (args, kwargs). Returns a mapping from
    # submodule name to that pair; submodules that never ran are absent. Any exception from
    # the model run is downgraded to a warning (partial results are returned). Hooks are
    # always removed afterwards. Does not work if two names refer to the same module object.
    """
    Generate inputs for targeting submodules in the given model. Note that if two submodules refer to the same obj, this
    function doesn't work.

    Args:
        model: root model.
        inputs: inputs to the root model.
        target_submodules: submodules that we want to generate inputs for.

    Returns:
        A dict that maps from submodule name to its inputs.
    """
    kwargs = kwargs or {}

    handles = []
    results = {}
    submodule_to_names = {mod: name for name, mod in model.named_modules()}

    def pre_forward(module, module_args, module_kwargs):
        # [FUNCTION DOC] HELPER: Forward pre-hook that stores the module's call args and kwargs in the
        # results mapping under the module's name (later calls overwrite earlier ones).
        results[submodule_to_names[module]] = (module_args, module_kwargs)

    try:
        for name, mod in model.named_modules():
            if name in target_submodules:
                handles.append(
                    mod.register_forward_pre_hook(pre_forward, with_kwargs=True)
                )
        model(*args, **kwargs)
    except Exception as e:
        warnings.warn(
            f"Failed to generate submodule inputs because of the following error:\n{e}",
            stacklevel=2,
        )
    finally:
        for h in handles:
            h.remove()
    return results


def report_exportability(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    strict: bool = True,
    pre_dispatch: bool = False,
) -> dict[str, Exception | None]:
    # [FUNCTION DOC] Try exporting mod (via the private export entry point with the given strict and
    # pre_dispatch flags); where a module fails, recurse into its direct children using inputs
    # captured by running the model once. Returns a mapping from dotted submodule name (empty
    # string for the root) to the raised exception, or None on success. Only the first module
    # of each type is tried, and modules without captured inputs are skipped (not in the report).
    # Side effects: logs usage telemetry, runs the model, logs successes and a de-duplicated
    # summary of failures.
    # KNOWN ISSUE: the de-duplication key splits the repr on a literal backslash-n sequence
    # rather than a real newline, so multi-line reprs are not truncated to the first line.
    """
    Report exportability issues for a module in one-shot.

    Args:
        mod: root module.
        args: args to the root module.
        kwargs: kwargs to the root module.
    Returns:
        A dict that maps from submodule name to the exception that was raised when trying to export it.
        `None` means the module is exportable without issue.
    Sample output:
        {
            '': UnsupportedOperatorException(func=<OpOverload(op='testlib.op_missing_meta', overload='default')>),
            'submod_1': UnsupportedOperatorException(func=<OpOverload(op='testlib.op_missing_meta', overload='default')>),
            'submod_2': None
        }
    """

    log_export_usage(event="export.report_exportability")

    kwargs = kwargs or {}

    all_submod_names = [name for name, _ in mod.named_modules() if name != ""]
    submod_inputs = _generate_inputs_for_submodules(mod, all_submod_names, args, kwargs)

    tried_module_types = set()
    report: dict[str, Exception | None] = {}

    def try_export(module, module_name, args, kwargs):
        # [FUNCTION DOC] HELPER: Recursive worker. Skips module types already tried; if inputs are known,
        # attempts an export and records None (and stops descending) on success, or the
        # exception (and continues into children) on failure. Writes into the enclosing report.
        nonlocal submod_inputs, report, strict, pre_dispatch, tried_module_types

        if type(module) in tried_module_types:
            return
        tried_module_types.add(type(module))

        if args is not None or kwargs is not None:
            try:
                torch.export._trace._export(
                    module,
                    args,
                    kwargs,
                    strict=strict,
                    pre_dispatch=pre_dispatch,
                )
                report[module_name] = None
                log.info("Successfully exported `%s`", module_name)
                return
            except Exception as e:
                short_msg = repr(e).split("\n")[0]
                log.warning(
                    "Failed exporting `%s` with exception: %s", module_name, short_msg
                )
                report[module_name] = e

        for name, submod in module.named_children():
            sub_module_name = name if module_name == "" else f"{module_name}.{name}"

            submod_args, submod_kwargs = submod_inputs.get(
                sub_module_name, (None, None)
            )

            try_export(submod, sub_module_name, submod_args, submod_kwargs)

        return

    try_export(mod, "", args, kwargs)

    unique_issues = set()
    for exception in report.values():
        if exception is not None:
            key = repr(exception).split("\\n")[0]
            unique_issues.add(key)

    log.warning("Found %d export issues:", len(unique_issues))
    for issue in unique_issues:
        log.warning(issue)

    return report
