# [PACKAGE DOC] Package for Inductor device-scoped heuristics: dispatches template (compile-time) and triton codegen (runtime) heuristics per device. This file only holds a docstring; no code or exports.
"""
Unified device-scoped heuristic module for PyTorch Inductor.

This module provides a single entry point for all device-specific heuristic
dispatch, covering both:

- template: pre-written template kernels (mm, conv, flex_attention...)
  used at compile time.
- triton_codegen: inductor-generated triton kernels (pointwise, reduction...)
  used at runtime.
"""
