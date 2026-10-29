# [PACKAGE DOC] Lookup-table package for Inductor template choices; exports only
# LookupTableChoices (from .choices), a choices handler that serves pre-configured
# template configs from a table. Importing it has no side effects beyond the import.
"""
Template lookup table system for PyTorch Inductor.

This package provides functionality for:
- Loading pre-configured template choices from lookup tables
- Managing template configurations and choices

All functionality is contained within the LookupTableChoices class.
You can customize any aspect by subclassing LookupTableChoices and overriding methods.

Usage:
    # Basic usage
    choices = LookupTableChoices()
    V.set_choices_handler(choices)

    # Custom usage
    class MyCustomChoices(LookupTableChoices):
        def _get_lookup_table(self):
            return my_custom_table

        def make_lookup_key(self, kernel_inputs, op_name, include_device=False):
            return f"custom_{op_name}_{hash(str(kernel_inputs))}"

    V.set_choices_handler(MyCustomChoices())
"""

from .choices import LookupTableChoices


__all__ = [
    "LookupTableChoices",
]
