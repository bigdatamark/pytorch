# [MODULE DOC] Defines TemplateConfigHeuristics, the base class that produces template configs (as params objects) for a kernel template.
# No import-time side effects.
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .params import DictKernelTemplateParams, KernelTemplateParams


if TYPE_CHECKING:
    from collections.abc import Generator

    from ...kernel_inputs import KernelInputs


class TemplateConfigHeuristics:
    # [CLASS DOC] Base class for generating sets of configs for a template. Stateless; subclasses override _get_template_configs_impl,
    # and optionally should_run, get_extra_kwargs and adjust_kernel_inputs.
    """Base class for generating sets of configs for an associated template."""

    def should_run(self, inputs: KernelInputs) -> bool:
        # [FUNCTION DOC] Gate hook: return whether this heuristic should produce configs at all. Base always returns True.
        """
        hookup to check whether the configs are right to run at all e.g. you can check
        max-autotune specific to your heuristic here or other things
        If this returns False, get_template_configs will yield no configs

        Args:
            inputs: KernelInputs
        """
        return True

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[KernelTemplateParams, None, None]:
        # [FUNCTION DOC] Yield a DictKernelTemplateParams for each config dict from _get_template_configs_impl, or nothing if should_run is false.
        # KNOWN ISSUE: an inline comment says extra_kwargs are fused into each config, but get_extra_kwargs is not called here; only the config dicts are wrapped.
        """
        Get template configs for the given inputs.

        Prefer to override the _get_template_configs_impl method
        to leverage things like should_run
        """
        if not self.should_run(kernel_inputs):
            return

        # Generate configs and fuse with extra_kwargs
        for config_dict in self._get_template_configs_impl(kernel_inputs, op_name):
            # Fuse extra_kwargs into config
            yield DictKernelTemplateParams(config_dict)

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        # [FUNCTION DOC] Main override point: yield config dicts for the inputs and op. Base yields nothing.
        """
        Get template configs for the given inputs.
        This is the main entry point for template-specific logic.
        """
        # base implementation yields no entries
        yield from []

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        # [FUNCTION DOC] Return kwargs constant across all configs of the template (base: empty dict).
        """
        Get extra kwargs for the given inputs/op for the template.

        Use this to return kwargs that are needed for the template, but
        do not change depending on the config/choice, but are rather
        always the same, for all configs
        """
        return {}

    def adjust_kernel_inputs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> KernelInputs:
        # [FUNCTION DOC] Hook to transform kernel inputs (for example squeeze/unsqueeze) before config generation. Base returns them unchanged.
        """
        Adjust kernel inputs for the given inputs/op for the template.

        override this to adjust the kernel inputs e.g. (un)squeezing
        """
        return kernel_inputs
